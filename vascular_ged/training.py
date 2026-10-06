from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .atomic import write_csv, write_json
from .graph import load_vtp


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class VascularPairDataset:
    def __init__(self, labels: str | Path, dataset_root: str | Path, expected_cost_config_id: str):
        import torch
        from torch_geometric.data import Data
        self.torch, self.Data = torch, Data
        self.root = Path(dataset_root).resolve()
        with Path(labels).open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError("label table is empty")
        patients: dict[str, str] = {}
        graph_paths: set[Path] = set()
        for row in self.rows:
            if row["cost_config_id"] != expected_cost_config_id:
                raise ValueError(f"cost configuration mismatch: {row['cost_config_id']}")
            lower, upper = float(row["lower_bound"]), float(row["upper_bound"])
            if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
                raise ValueError(f"invalid bounds for pair {row['pair_id']}")
            for suffix in ("a", "b"):
                patient = row[f"patient_{suffix}"]
                if patient in patients and patients[patient] != row["greed_split"]:
                    raise ValueError(f"patient {patient} crosses partitions")
                patients[patient] = row["greed_split"]
                path = (self.root / row[f"graph_{suffix}_relative_path"]).resolve()
                if self.root not in path.parents or not path.is_file():
                    raise FileNotFoundError(path)
                graph_paths.add(path)
        for path in sorted(graph_paths):
            graph = load_vtp(path)
            if not graph.num_nodes:
                raise ValueError(f"empty graph is unsupported by the initial GIN: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _graph(self, relative: str):
        graph = load_vtp(self.root / relative)
        if not graph.num_nodes:
            raise ValueError("empty graphs are excluded from the initial GREED experiment")
        edge_index = self.torch.as_tensor(graph.edges.T.copy(), dtype=self.torch.long)
        edge_index = self.torch.cat((edge_index, edge_index.flip(0)), dim=1)
        return self.Data(x=self.torch.as_tensor(graph.coordinates, dtype=self.torch.float32), edge_index=edge_index)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return (
            self._graph(row["graph_a_relative_path"]), self._graph(row["graph_b_relative_path"]),
            self.torch.tensor(float(row["lower_bound"]), dtype=self.torch.float32),
            self.torch.tensor(float(row["upper_bound"]), dtype=self.torch.float32), index,
        )


def make_model(layers: int = 8, hidden: int = 64, embedding: int = 64):
    import torch
    from torch_geometric.nn import GINConv, global_add_pool

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Linear(3, hidden)
            self.layers = torch.nn.ModuleList([
                GINConv(torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)))
                for _ in range(layers)
            ])
            self.output = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, embedding))

        def forward(self, graph):
            x = torch.relu(self.input(graph.x))
            for layer in self.layers:
                x = torch.relu(layer(x, graph.edge_index))
            return self.output(global_add_pool(x, graph.batch))

    class GeometricGREED(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()

        def embed(self, graph):
            return self.encoder(graph)

        def forward(self, graph_a, graph_b):
            return torch.linalg.vector_norm(self.embed(graph_a) - self.embed(graph_b), dim=-1)

        @staticmethod
        def interval_loss(prediction, lower, upper):
            return (torch.relu(lower - prediction).square() + torch.relu(prediction - upper).square()).mean()

    return GeometricGREED()


def _rank_metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    try:
        from scipy.stats import kendalltau, spearmanr
        return float(spearmanr(target, prediction).statistic), float(kendalltau(target, prediction).statistic)
    except ImportError:
        return math.nan, math.nan


def train(config_path: str | Path) -> None:
    import torch
    from torch_geometric.loader import DataLoader
    def expand(value):
        if isinstance(value, str): return os.path.expandvars(value)
        if isinstance(value, list): return [expand(item) for item in value]
        if isinstance(value, dict): return {key: expand(item) for key, item in value.items()}
        return value
    cfg = expand(yaml.safe_load(Path(config_path).read_text()))
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    train_set = VascularPairDataset(cfg["train_labels"], cfg["dataset_root"], cfg["cost_config_id"])
    val_set = VascularPairDataset(cfg["validation_labels"], cfg["dataset_root"], cfg["cost_config_id"])
    if len(train_set) != 8000 or len(val_set) != 2000:
        raise ValueError(f"expected 8000/2000 train/validation pairs, got {len(train_set)}/{len(val_set)}")
    train_patients = {row[key] for row in train_set.rows for key in ("patient_a", "patient_b")}
    val_patients = {row[key] for row in val_set.rows for key in ("patient_a", "patient_b")}
    if train_patients & val_patients:
        raise ValueError("GREED training and validation patients overlap")
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"])
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["workers"])
    model = make_model(cfg["model"]["layers"], cfg["model"]["hidden_dim"], cfg["model"]["embedding_dim"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    history, best, best_metrics, best_group_metrics = [], math.inf, {}, {}
    for epoch in range(1, cfg["epochs"] + 1):
        tic = time.perf_counter(); model.train(); train_loss = []
        for ga, gb, lower, upper, _ in train_loader:
            pred = model(ga, gb); loss = model.interval_loss(pred, lower, upper)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); train_loss.append(loss.item())
        model.eval(); val_loss = []; predictions = []; lowers = []; uppers = []; indices = []
        with torch.no_grad():
            for ga, gb, lower, upper, index in val_loader:
                pred = model(ga, gb); val_loss.append(model.interval_loss(pred, lower, upper).item())
                predictions.extend(pred.tolist()); lowers.extend(lower.tolist()); uppers.extend(upper.tolist()); indices.extend(index.tolist())
        current = float(np.mean(val_loss)); exact_mask = np.isclose(lowers, uppers)
        exact_target = np.asarray(lowers)[exact_mask]; exact_pred = np.asarray(predictions)[exact_mask]
        mae = float(np.mean(np.abs(exact_target - exact_pred))) if len(exact_target) else math.nan
        rmse = float(np.sqrt(np.mean((exact_target - exact_pred) ** 2))) if len(exact_target) else math.nan
        midpoint = (np.asarray(lowers) + np.asarray(uppers)) / 2
        spearman, kendall = _rank_metrics(midpoint, np.asarray(predictions))
        history.append({"epoch": epoch, "training_loss": np.mean(train_loss), "validation_interval_loss": current,
                        "validation_exact_mae": mae, "validation_exact_rmse": rmse, "spearman": spearman,
                        "kendall": kendall, "runtime_seconds": time.perf_counter() - tic})
        state = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "config": cfg}
        torch.save(state, output / "latest_checkpoint.pt")
        if current < best:
            best = current; torch.save(state, output / "best_model.pt")
            pred_rows = []
            for i, prediction, lower, upper in zip(indices, predictions, lowers, uppers):
                pred_rows.append(dict(val_set.rows[i], prediction=prediction,
                                      interval_violation=max(lower - prediction, 0) + max(prediction - upper, 0),
                                      absolute_error=(abs(prediction - lower) if math.isclose(lower, upper) else "")))
            write_csv(output / "validation_predictions.csv", pred_rows, list(pred_rows[0]))
            best_metrics = dict(history[-1])
            best_group_metrics = {}
            for key in sorted({row["size_stratum"] for row in pred_rows}):
                group = [row for row in pred_rows if row["size_stratum"] == key]
                best_group_metrics[f"size_stratum:{key}"] = float(np.mean([float(row["interval_violation"]) for row in group]))
            cuts = np.quantile(midpoint, [0, 1/3, 2/3, 1])
            for bin_index in range(3):
                mask = (midpoint >= cuts[bin_index]) & (midpoint <= cuts[bin_index + 1] if bin_index == 2 else midpoint < cuts[bin_index + 1])
                best_group_metrics[f"ged_range:{bin_index}:{cuts[bin_index]:.6g}-{cuts[bin_index+1]:.6g}"] = float(np.mean(np.asarray([r["interval_violation"] for r in pred_rows], dtype=float)[mask]))
        write_csv(output / "training_history.csv", history, list(history[0]))
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    metadata = {
        "train_labels_sha256": sha256(cfg["train_labels"]),
        "validation_labels_sha256": sha256(cfg["validation_labels"]),
        "cost_config_sha256": sha256(cfg["cost_config_path"]), "patient_split_sha256": sha256(cfg["patient_split"]),
        "git_commit": os.popen("git rev-parse HEAD").read().strip(), "best_validation_interval_loss": best,
        "best_metrics": best_metrics, "validation_group_metrics": best_group_metrics,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"), "slurm_cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK"),
        "peak_memory_bytes": __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss * 1024,
    }
    write_json(output / "model_metadata.json", metadata)
    report = ["# GREED 3D CPU pilot", "", f"Best validation interval loss: {best:.8g}",
              f"Best epoch metrics: `{best_metrics}`", f"Grouped validation interval violations: `{best_group_metrics}`",
              f"Peak memory bytes: {metadata['peak_memory_bytes']}", ""]
    reports = Path(cfg["output_dir"]).parents[1] / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "greed_pilot_report.md").write_text("\n".join(report), encoding="utf-8")
