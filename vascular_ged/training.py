from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from .atomic import write_csv, write_json
from .graph import load_vtp


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _balanced_pair_subset(
    rows: list[dict[str, str]], count: int | None, seed: int,
) -> list[dict[str, str]]:
    if count is None or count == len(rows):
        return rows
    if count <= 0 or count > len(rows):
        raise ValueError(f"requested training subset has {count} rows, available rows: {len(rows)}")
    cells = sorted({(row["sampling_strategy"], row["size_stratum"]) for row in rows})
    if len(cells) != 9:
        raise ValueError(f"expected nine strategy/stratum cells, found {len(cells)}")
    quotas = {cell: count // len(cells) + (index < count % len(cells)) for index, cell in enumerate(cells)}
    selected_ids: set[str] = set()
    for cell in cells:
        candidates = [row for row in rows if (row["sampling_strategy"], row["size_stratum"]) == cell]
        ranked = sorted(
            candidates,
            key=lambda row: hashlib.sha256(f"{seed}\0{row['pair_id']}".encode()).digest(),
        )
        quota = quotas[cell]
        if len(ranked) < quota:
            raise ValueError(f"cell {cell} has only {len(ranked)} rows, requested {quota}")
        selected_ids.update(row["pair_id"] for row in ranked[:quota])
    selected = [row for row in rows if row["pair_id"] in selected_ids]
    if len(selected) != count:
        raise AssertionError(f"balanced subset produced {len(selected)} rows, expected {count}")
    return selected


class VascularPairDataset:
    def __init__(
        self, labels: str | Path, dataset_root: str | Path,
        expected_cost_config_id: str, expected_split: str | None = None,
        pair_count: int | None = None, subset_seed: int = 0,
    ):
        import torch
        from torch_geometric.data import Data
        self.torch, self.Data = torch, Data
        self.root = Path(dataset_root).resolve()
        with Path(labels).open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError("label table is empty")
        self.rows = _balanced_pair_subset(self.rows, pair_count, subset_seed)
        pair_ids = [row["pair_id"] for row in self.rows]
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("label table contains duplicate pair IDs")
        patients: dict[str, str] = {}
        graph_rows: dict[str, list[tuple[dict[str, str], str]]] = {}
        for row in self.rows:
            if row["cost_config_id"] != expected_cost_config_id:
                raise ValueError(f"cost configuration mismatch: {row['cost_config_id']}")
            if row.get("solver_status") not in {"ok", "analytic_empty_graph"}:
                raise ValueError(f"unsuccessful label row {row['pair_id']}")
            if expected_split is not None and row["greed_split"] != expected_split:
                raise ValueError(f"pair {row['pair_id']} is in the wrong GREED partition")
            lower, upper = float(row["lower_bound"]), float(row["upper_bound"])
            if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
                raise ValueError(f"invalid bounds for pair {row['pair_id']}")
            if row["patient_a"] == row["patient_b"]:
                raise ValueError(f"same-patient pair is not allowed: {row['pair_id']}")
            for suffix in ("a", "b"):
                patient = row[f"patient_{suffix}"]
                if patient in patients and patients[patient] != row["greed_split"]:
                    raise ValueError(f"patient {patient} crosses partitions")
                patients[patient] = row["greed_split"]
                path = (self.root / row[f"graph_{suffix}_relative_path"]).resolve()
                if self.root not in path.parents or not path.is_file():
                    raise FileNotFoundError(path)
                relative = row[f"graph_{suffix}_relative_path"]
                graph_rows.setdefault(relative, []).append((row, suffix))
        self.graphs = {}
        for relative, references in sorted(graph_rows.items()):
            graph = load_vtp(self.root / relative)
            if not graph.num_nodes:
                raise ValueError(f"empty graph is unsupported by the initial GIN: {relative}")
            for row, suffix in references:
                if graph.num_nodes != int(row[f"node_count_{suffix}"]):
                    raise ValueError(f"node count mismatch for {relative}")
                if graph.num_edges != int(row[f"edge_count_{suffix}"]):
                    raise ValueError(f"edge count mismatch for {relative}")
            edge_index = self.torch.as_tensor(graph.edges.T.copy(), dtype=self.torch.long)
            edge_index = self.torch.cat((edge_index, edge_index.flip(0)), dim=1)
            self.graphs[relative] = self.Data(
                x=self.torch.as_tensor(graph.coordinates, dtype=self.torch.float32),
                edge_index=edge_index,
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _graph(self, relative: str):
        return self.graphs[relative]

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
            self.output = torch.nn.Sequential(
                torch.nn.Linear(hidden * (layers + 1), hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, embedding),
            )

        def forward(self, graph):
            x = torch.relu(self.input(graph.x))
            representations = [x]
            residual = x
            for index, layer in enumerate(self.layers):
                x = layer(x, graph.edge_index)
                if index % 2 == 1:
                    x = x + residual
                    residual = x
                x = torch.relu(x)
                representations.append(x)
            concatenated = torch.cat(representations, dim=-1)
            return self.output(global_add_pool(concatenated, graph.batch))

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
    if len(target) < 2 or np.ptp(target) == 0 or np.ptp(prediction) == 0:
        return math.nan, math.nan
    try:
        from scipy.stats import kendalltau, spearmanr
        return float(spearmanr(target, prediction).statistic), float(kendalltau(target, prediction).statistic)
    except ImportError:
        return math.nan, math.nan


def _expand(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _patient_assignment(path: str | Path) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("patient split is empty")
    if any(row["relationformer_split"] != "train" for row in rows):
        raise ValueError("patient split contains non-training RelationFormer patients")
    assignment = {row["patient_id"]: row["greed_split"] for row in rows}
    if len(assignment) != len(rows):
        raise ValueError("patient split contains duplicate patient IDs")
    if set(assignment.values()) - {"train", "validation"}:
        raise ValueError("patient split contains an invalid GREED partition")
    return assignment


def _validate_patient_assignment(
    train_set: VascularPairDataset, val_set: VascularPairDataset, assignment: dict[str, str],
) -> tuple[set[str], set[str]]:
    train_patients = {row[key] for row in train_set.rows for key in ("patient_a", "patient_b")}
    val_patients = {row[key] for row in val_set.rows for key in ("patient_a", "patient_b")}
    if train_patients & val_patients:
        raise ValueError("GREED training and validation patients overlap")
    for expected, patients in (("train", train_patients), ("validation", val_patients)):
        wrong = {patient for patient in patients if assignment.get(patient) != expected}
        if wrong:
            raise ValueError(f"{len(wrong)} label patients disagree with the authoritative {expected} split")
    return train_patients, val_patients


def _atomic_torch_save(torch, value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _evaluation_metrics(
    predictions: np.ndarray, lower: np.ndarray, upper: np.ndarray,
) -> dict[str, float | int]:
    if not (np.isfinite(predictions).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError("evaluation inputs contain non-finite values")
    below = np.maximum(lower - predictions, 0)
    above = np.maximum(predictions - upper, 0)
    violation = below + above
    exact = np.isclose(lower, upper, atol=1e-9)
    bounded = ~exact
    exact_target, exact_prediction = lower[exact], predictions[exact]
    spearman, kendall = _rank_metrics(exact_target, exact_prediction) if exact.any() else (math.nan, math.nan)
    return {
        "interval_loss": float(np.mean(below ** 2 + above ** 2)),
        "interval_coverage": float(np.mean(violation == 0)),
        "mean_interval_violation": float(np.mean(violation)),
        "exact_count": int(exact.sum()),
        "bounded_count": int(bounded.sum()),
        "bounded_interval_coverage": float(np.mean(violation[bounded] == 0)) if bounded.any() else math.nan,
        "bounded_mean_interval_violation": float(np.mean(violation[bounded])) if bounded.any() else math.nan,
        "exact_mae": float(np.mean(np.abs(exact_target - exact_prediction))) if exact.any() else math.nan,
        "exact_rmse": float(np.sqrt(np.mean((exact_target - exact_prediction) ** 2))) if exact.any() else math.nan,
        "exact_spearman": spearman,
        "exact_kendall": kendall,
    }


def _initialize_continuation(torch, source: Path, output: Path, cfg: dict, input_hashes: dict) -> None:
    """Fork a capped run; only the epoch budget and output locations may change."""
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('continuation must use a separate output directory')
    state = torch.load(source / 'latest_checkpoint.pt', map_location='cpu', weights_only=False)
    original = state['config']
    expected = dict(original, epochs=cfg['epochs'], output_dir=cfg['output_dir'], report_path=cfg['report_path'])
    if cfg != expected or state['input_hashes'] != input_hashes:
        raise ValueError('continuation configuration or input hashes do not match the source')
    if int(cfg['epochs']) <= int(original['epochs']) or int(state['epoch']) != int(original['epochs']):
        raise ValueError('continuation requires a capped source and a larger epoch budget')
    patience = int(cfg.get('early_stopping_patience', 0))
    if patience and state['epochs_without_improvement'] >= patience:
        raise ValueError('source already satisfied early stopping')
    best = torch.load(source / 'best_model.pt', map_location='cpu', weights_only=False)
    if best['input_hashes'] != input_hashes or best['config'] != original or best['epoch'] != state['best_metrics']['epoch']:
        raise ValueError('source best checkpoint does not match the latest checkpoint')
    prediction_text = (source / 'validation_predictions.csv').read_text(encoding='utf-8')
    if any(output.iterdir()):
        raise FileExistsError('new continuation directory must be empty')
    for checkpoint in (state, best):
        checkpoint['config'] = cfg
    _atomic_torch_save(torch, best, output / 'best_model.pt')
    _atomic_text(output / 'validation_predictions.csv', prediction_text)
    write_json(output / 'continuation.json', {
        'source_directory': str(source), 'source_checkpoint_sha256': sha256(source / 'latest_checkpoint.pt'),
        'source_epoch': state['epoch'], 'new_epoch_limit': cfg['epochs'],
    })
    # Commit the latest checkpoint last so a partially initialized fork cannot resume.
    _atomic_torch_save(torch, state, output / 'latest_checkpoint.pt')


def _constant_baseline(train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]) -> dict[str, float]:
    exact_train = [float(row["lower_bound"]) for row in train_rows if row["is_exact"].lower() in {"true", "1"}]
    if not exact_train:
        raise ValueError("training labels contain no exact rows for the constant baseline")
    prediction = float(np.median(exact_train))
    lower = np.asarray([float(row["lower_bound"]) for row in val_rows])
    upper = np.asarray([float(row["upper_bound"]) for row in val_rows])
    metrics = _evaluation_metrics(np.full(len(val_rows), prediction), lower, upper)
    # Rank correlation is undefined for a constant prediction.
    metrics.pop("exact_spearman")
    metrics.pop("exact_kendall")
    return {"constant_prediction": prediction, **metrics}


def train(
    config_path: str | Path, preflight_only: bool = False, *, seed: int | None = None,
    train_count: int | None = None, output_dir: str | Path | None = None,
    train_labels: str | None = None, validation_labels: str | None = None, patient_split: str | None = None,
    epochs: int | None = None, resume_from: str | Path | None = None,
) -> None:
    import torch
    from torch_geometric.loader import DataLoader
    run_started = time.perf_counter()
    if preflight_only and resume_from is not None:
        raise ValueError('continuation cannot be combined with preflight-only')
    cfg = _expand(yaml.safe_load(Path(config_path).read_text()))
    if seed is not None:
        cfg["seed"] = seed
    if epochs is not None:
        if epochs <= 0:
            raise ValueError('epochs must be positive')
        cfg['epochs'] = epochs
    if train_count is not None:
        cfg["train_count"] = train_count
    for key, value in (("train_labels", train_labels), ("validation_labels", validation_labels), ("patient_split", patient_split)):
        if value is not None:
            cfg[key] = value
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)
        cfg["report_path"] = str(Path(output_dir) / "training_report.md")
    required_paths = ("dataset_root", "train_labels", "validation_labels", "patient_split", "cost_config_path")
    unresolved = [key for key in required_paths if "$" in str(cfg[key])]
    if unresolved:
        raise ValueError(f"unresolved environment variables in training config: {unresolved}")
    if cfg["model"].get("input_dim") != 3 or cfg["model"].get("pooling") != "sum":
        raise ValueError("the initial vascular GREED model requires input_dim=3 and sum pooling")
    if cfg.get("checkpoint_metric") != "validation_interval_loss":
        raise ValueError("checkpoint_metric must be validation_interval_loss")
    if cfg.get("augmentation") != "none":
        raise ValueError("the initial vascular GREED experiment requires canonical unaugmented coordinates")
    cost_config = yaml.safe_load(Path(cfg["cost_config_path"]).read_text(encoding="utf-8"))
    if cost_config.get("id") != cfg["cost_config_id"]:
        raise ValueError("training and cost configuration IDs disagree")
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    torch.set_num_threads(int(cfg.get("torch_threads", os.getenv("SLURM_CPUS_PER_TASK", "1"))))
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        "train_labels_sha256": sha256(cfg["train_labels"]),
        "validation_labels_sha256": sha256(cfg["validation_labels"]),
        "cost_config_sha256": sha256(cfg["cost_config_path"]),
        "patient_split_sha256": sha256(cfg["patient_split"]),
    }
    train_set = VascularPairDataset(
        cfg["train_labels"], cfg["dataset_root"], cfg["cost_config_id"], "train",
        int(cfg["train_count"]), int(cfg["subset_seed"]),
    )
    val_set = VascularPairDataset(
        cfg["validation_labels"], cfg["dataset_root"], cfg["cost_config_id"], "validation",
    )
    input_hashes.update({
        "train_subset_pair_ids_sha256": hashlib.sha256(
            "\n".join(row["pair_id"] for row in train_set.rows).encode()
        ).hexdigest(),
        "validation_pair_ids_sha256": hashlib.sha256(
            "\n".join(row["pair_id"] for row in val_set.rows).encode()
        ).hexdigest(),
    })
    if len(train_set) != int(cfg["train_count"]) or len(val_set) != int(cfg["validation_count"]):
        raise ValueError(
            f"expected {cfg['train_count']}/{cfg['validation_count']} train/validation pairs, "
            f"got {len(train_set)}/{len(val_set)}"
        )
    assignment = _patient_assignment(cfg["patient_split"])
    train_patients, val_patients = _validate_patient_assignment(train_set, val_set, assignment)
    split_train_count = sum(split == "train" for split in assignment.values())
    split_validation_count = sum(split == "validation" for split in assignment.values())
    workers = int(cfg["workers"])
    loader_options = {"batch_size": int(cfg["batch_size"]), "num_workers": workers}
    if workers:
        loader_options["persistent_workers"] = True
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    model = make_model(cfg["model"]["layers"], cfg["model"]["hidden_dim"], cfg["model"]["embedding_dim"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    baseline = _constant_baseline(train_set.rows, val_set.rows)
    dataset_ready_seconds = time.perf_counter() - run_started
    print(json.dumps({
        "stage": "dataset_ready", "train_pairs": len(train_set), "validation_pairs": len(val_set),
        "train_graphs": len(train_set.graphs), "validation_graphs": len(val_set.graphs),
        "train_patients": len(train_patients), "validation_patients": len(val_patients),
        "split_train_patients": split_train_count, "split_validation_patients": split_validation_count,
        "model_parameters": parameter_count, "constant_baseline": baseline,
        "dataset_ready_seconds": dataset_ready_seconds,
    }, sort_keys=True), flush=True)

    if preflight_only:
        ga, gb, lower, upper, _ = next(iter(train_loader))
        prediction = model(ga, gb)
        loss = model.interval_loss(prediction, lower, upper)
        if not torch.isfinite(loss):
            raise FloatingPointError("preflight produced a non-finite loss")
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
        optimizer.step()
        checkpoint = output / ".preflight_checkpoint.pt"
        _atomic_torch_save(torch, model.state_dict(), checkpoint)
        reloaded = make_model(cfg["model"]["layers"], cfg["model"]["hidden_dim"], cfg["model"]["embedding_dim"])
        reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        checkpoint.unlink()
        result = {
            "status": "passed", "batch_size": int(lower.numel()), "batch_loss": float(loss.item()),
            "train_pairs": len(train_set), "validation_pairs": len(val_set),
            "train_graphs": len(train_set.graphs), "validation_graphs": len(val_set.graphs),
            "train_patients": len(train_patients), "validation_patients": len(val_patients),
            "split_train_patients": split_train_count, "split_validation_patients": split_validation_count,
            "model_parameters": parameter_count, "dataset_ready_seconds": dataset_ready_seconds,
            "preflight_runtime_seconds": time.perf_counter() - run_started,
            "peak_memory_bytes": __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss * 1024,
            **input_hashes,
        }
        write_json(output / "preflight.json", result)
        print(json.dumps({"stage": "preflight_complete", **result}, sort_keys=True), flush=True)
        return

    history: list[dict[str, object]] = []
    best, best_metrics, best_group_metrics = math.inf, {}, {}
    epochs_without_improvement = 0
    start_epoch = 1
    latest = output / "latest_checkpoint.pt"
    if resume_from is not None:
        if preflight_only or not bool(cfg.get('resume', True)):
            raise ValueError('continuation requires resume enabled and a training run')
        if latest.exists():
            provenance = json.loads((output / 'continuation.json').read_text())
            if provenance['source_directory'] != str(Path(resume_from).resolve()):
                raise ValueError('continuation source changed')
        else:
            _initialize_continuation(torch, Path(resume_from), output, cfg, input_hashes)
    if bool(cfg.get("resume", True)) and latest.is_file():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        if state.get("config") != cfg or state.get("input_hashes") != input_hashes:
            raise ValueError("existing checkpoint configuration or input hashes do not match this run")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        history = state["history"]
        best = float(state["best_validation_interval_loss"])
        best_metrics = state["best_metrics"]
        best_group_metrics = state["best_group_metrics"]
        epochs_without_improvement = int(state["epochs_without_improvement"])
        start_epoch = int(state["epoch"]) + 1
        torch.set_rng_state(state["torch_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        print(json.dumps({"stage": "resumed", "start_epoch": start_epoch, "best": best}), flush=True)

    patience = int(cfg.get("early_stopping_patience", 0))
    stopped_early = bool(patience and epochs_without_improvement >= patience)
    epoch_range = range(start_epoch, int(cfg["epochs"]) + 1) if not stopped_early else ()
    for epoch in epoch_range:
        tic = time.perf_counter(); model.train(); train_loss_sum = 0.0; train_count = 0
        for ga, gb, lower, upper, _ in train_loader:
            pred = model(ga, gb); loss = model.interval_loss(pred, lower, upper)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"epoch {epoch} produced a non-finite training loss")
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
            optimizer.step()
            train_loss_sum += float(loss.item()) * int(lower.numel()); train_count += int(lower.numel())
        model.eval(); predictions = []; lowers = []; uppers = []; indices = []
        with torch.no_grad():
            for ga, gb, lower, upper, index in val_loader:
                pred = model(ga, gb)
                predictions.extend(pred.tolist()); lowers.extend(lower.tolist()); uppers.extend(upper.tolist()); indices.extend(index.tolist())
        prediction_array = np.asarray(predictions)
        lower_array, upper_array = np.asarray(lowers), np.asarray(uppers)
        metrics = _evaluation_metrics(prediction_array, lower_array, upper_array)
        current = metrics["interval_loss"]
        epoch_metrics = {
            "epoch": epoch, "training_loss": train_loss_sum / train_count,
            "validation_interval_loss": current,
            "validation_interval_coverage": metrics["interval_coverage"],
            "validation_mean_interval_violation": metrics["mean_interval_violation"],
            "validation_bounded_interval_coverage": metrics["bounded_interval_coverage"],
            "validation_bounded_mean_interval_violation": metrics["bounded_mean_interval_violation"],
            "validation_exact_mae": metrics["exact_mae"],
            "validation_exact_rmse": metrics["exact_rmse"],
            "validation_exact_spearman": metrics["exact_spearman"],
            "validation_exact_kendall": metrics["exact_kendall"],
            "runtime_seconds": time.perf_counter() - tic,
        }
        history.append(epoch_metrics)
        improved = current < best - float(cfg.get("early_stopping_min_delta", 0.0))
        if improved:
            best = current; epochs_without_improvement = 0
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
            midpoint = (lower_array + upper_array) / 2
            cuts = np.quantile(midpoint, [0, 1/3, 2/3, 1])
            for bin_index in range(3):
                mask = (midpoint >= cuts[bin_index]) & (midpoint <= cuts[bin_index + 1] if bin_index == 2 else midpoint < cuts[bin_index + 1])
                if mask.any():
                    best_group_metrics[f"bound_midpoint_range:{bin_index}:{cuts[bin_index]:.6g}-{cuts[bin_index+1]:.6g}"] = float(np.mean(np.asarray([r["interval_violation"] for r in pred_rows], dtype=float)[mask]))
        else:
            epochs_without_improvement += 1
        state = {
            "epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "config": cfg, "input_hashes": input_hashes, "history": history,
            "best_validation_interval_loss": best, "best_metrics": best_metrics,
            "best_group_metrics": best_group_metrics,
            "epochs_without_improvement": epochs_without_improvement,
            "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
        }
        _atomic_torch_save(torch, state, latest)
        if improved:
            _atomic_torch_save(torch, state, output / "best_model.pt")
        write_csv(output / "training_history.csv", history, list(history[0]))
        print(json.dumps({"stage": "epoch", **epoch_metrics, "best": best}, sort_keys=True), flush=True)
        if patience and epochs_without_improvement >= patience:
            stopped_early = True
            print(json.dumps({"stage": "early_stopping", "epoch": epoch, "patience": patience}), flush=True)
            break
    _atomic_text(output / "resolved_config.yaml", yaml.safe_dump(cfg, sort_keys=True))
    metadata = {
        **input_hashes,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "best_validation_interval_loss": best,
        "best_metrics": best_metrics, "validation_group_metrics": best_group_metrics,
        "constant_validation_baseline": baseline,
        "epochs_completed": len(history), "stopped_early": stopped_early,
        "train_pair_count": len(train_set), "validation_pair_count": len(val_set),
        "training_seed": int(cfg["seed"]), "training_subset_seed": int(cfg["subset_seed"]),
        "continuation_source": str(Path(resume_from).resolve()) if resume_from is not None else None,
        "train_unique_graph_count": len(train_set.graphs), "validation_unique_graph_count": len(val_set.graphs),
        "train_patient_count": len(train_patients), "validation_patient_count": len(val_patients),
        "split_train_patient_count": split_train_count, "split_validation_patient_count": split_validation_count,
        "model_parameter_count": parameter_count,
        "dataset_ready_seconds": dataset_ready_seconds,
        "total_job_runtime_seconds": time.perf_counter() - run_started,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"), "slurm_cpus_per_task": os.getenv("SLURM_CPUS_PER_TASK"),
        "peak_memory_bytes": __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss * 1024,
    }
    write_json(output / "model_metadata.json", metadata)
    report = ["# GREED 3D CPU pilot", "", f"Best validation interval loss: {best:.8g}",
              f"Constant validation baseline: `{baseline}`",
              f"Best epoch metrics: `{best_metrics}`", f"Grouped validation interval violations: `{best_group_metrics}`",
              f"Peak memory bytes: {metadata['peak_memory_bytes']}", ""]
    report_path = Path(cfg.get("report_path", output / "training_report.md"))
    _atomic_text(report_path, "\n".join(report))


def analyze_learning_curve(
    experiment_root: str | Path, report_root: str | Path,
    expected_counts: tuple[int, ...] = (1000, 2000, 4000, 8000),
    expected_seeds: tuple[int, ...] = (161803, 271828, 314159),
) -> dict[str, object]:
    experiment_root, report_root = Path(experiment_root), Path(report_root)
    metadata_paths = sorted(experiment_root.glob("pairs_*/seed_*/model_metadata.json"))
    runs: list[dict[str, object]] = []
    for path in metadata_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        best = metadata["best_metrics"]
        runs.append({
            "train_pair_count": int(metadata["train_pair_count"]),
            "training_seed": int(metadata["training_seed"]),
            "best_epoch": int(best["epoch"]),
            "validation_interval_loss": float(best["validation_interval_loss"]),
            "validation_mean_interval_violation": float(best["validation_mean_interval_violation"]),
            "validation_bounded_interval_coverage": float(best["validation_bounded_interval_coverage"]),
            "validation_bounded_mean_interval_violation": float(best["validation_bounded_mean_interval_violation"]),
            "validation_exact_mae": float(best["validation_exact_mae"]),
            "validation_exact_rmse": float(best["validation_exact_rmse"]),
            "validation_exact_spearman": float(best["validation_exact_spearman"]),
            "validation_exact_kendall": float(best["validation_exact_kendall"]),
            "epochs_completed": int(metadata["epochs_completed"]),
            "total_job_runtime_seconds": float(metadata["total_job_runtime_seconds"]),
            "peak_memory_bytes": int(metadata["peak_memory_bytes"]),
            "slurm_job_id": metadata.get("slurm_job_id", ""),
            "train_subset_pair_ids_sha256": metadata["train_subset_pair_ids_sha256"],
            "validation_pair_ids_sha256": metadata["validation_pair_ids_sha256"],
            "git_commit": metadata["git_commit"],
        })
    keys = [(row["train_pair_count"], row["training_seed"]) for row in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("learning-curve results contain duplicate pair-count/seed runs")
    actual_counts = {int(row["train_pair_count"]) for row in runs}
    if actual_counts != set(expected_counts):
        raise ValueError(f"learning curve has pair counts {sorted(actual_counts)}, expected {list(expected_counts)}")
    validation_hashes = {row["validation_pair_ids_sha256"] for row in runs}
    if len(validation_hashes) != 1:
        raise ValueError("learning-curve runs do not share one validation set")
    commits = {row["git_commit"] for row in runs}
    if len(commits) != 1:
        raise ValueError("learning-curve runs were produced by different Git commits")
    aggregates: dict[str, dict[str, float | int]] = {}
    metric_names = (
        "validation_interval_loss", "validation_mean_interval_violation",
        "validation_bounded_interval_coverage", "validation_bounded_mean_interval_violation",
        "validation_exact_mae", "validation_exact_rmse",
        "validation_exact_spearman", "validation_exact_kendall",
    )
    for count in expected_counts:
        group = [row for row in runs if row["train_pair_count"] == count]
        actual_seeds = {int(row["training_seed"]) for row in group}
        if actual_seeds != set(expected_seeds):
            raise ValueError(f"pair count {count} has seeds {sorted(actual_seeds)}, expected {list(expected_seeds)}")
        subset_hashes = {row["train_subset_pair_ids_sha256"] for row in group}
        if len(subset_hashes) != 1:
            raise ValueError(f"{count}-pair runs do not share one training subset")
        summary: dict[str, float | int] = {"runs": len(group)}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group])
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite {metric} in {count}-pair runs")
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
        aggregates[str(count)] = summary
    report_root.mkdir(parents=True, exist_ok=True)
    write_csv(report_root / "greed_learning_curve_runs.csv", runs, list(runs[0]))
    result = {
        "experiment_root": str(experiment_root.resolve()), "runs": len(runs),
        "git_commit": next(iter(commits)),
        "validation_pair_ids_sha256": next(iter(validation_hashes)), "aggregates": aggregates,
    }
    write_json(report_root / "greed_learning_curve_metrics.json", result)
    lines = [
        "# GREED 3D learning curve", "",
        "All runs use the same 2,000-pair patient-disjoint validation set.", "",
        "| Training pairs | Interval loss | Exact MAE | Exact RMSE | Spearman | Bounded coverage |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for count in expected_counts:
        item = aggregates[str(count)]
        lines.append(
            f"| {count} | {item['validation_interval_loss_mean']:.4f} ± {item['validation_interval_loss_std']:.4f} "
            f"| {item['validation_exact_mae_mean']:.4f} ± {item['validation_exact_mae_std']:.4f} "
            f"| {item['validation_exact_rmse_mean']:.4f} ± {item['validation_exact_rmse_std']:.4f} "
            f"| {item['validation_exact_spearman_mean']:.4f} ± {item['validation_exact_spearman_std']:.4f} "
            f"| {item['validation_bounded_interval_coverage_mean']:.4f} ± {item['validation_bounded_interval_coverage_std']:.4f} |"
        )
    _atomic_text(report_root / "greed_learning_curve_report.md", "\n".join(lines) + "\n")
    return result


def analyze_50k(experiment_root: str | Path, baseline_root: str | Path, report_root: str | Path) -> dict:
    experiment_root, baseline_root, report_root = Path(experiment_root), Path(baseline_root), Path(report_root)
    seeds = (161803, 271828, 314159)
    compared_settings = ('model', 'batch_size', 'epochs', 'learning_rate', 'weight_decay',
                         'max_grad_norm', 'checkpoint_metric', 'early_stopping_patience',
                         'early_stopping_min_delta', 'augmentation', 'cost_config_id')
    baseline_runs = []
    for seed in seeds:
        old_dir = baseline_root / f'pairs_8000/seed_{seed}'
        new_dir = experiment_root / f'pairs_50000/seed_{seed}'
        old = json.loads((old_dir / 'model_metadata.json').read_text())
        new = json.loads((new_dir / 'model_metadata.json').read_text())
        if old['train_pair_count'] != 8000 or new['train_pair_count'] != 50000:
            raise ValueError('comparison requires 8k and 50k training runs')
        for metadata in (old, new):
            if metadata['training_seed'] != seed or metadata['validation_pair_count'] != 2000:
                raise ValueError('comparison seed or validation count mismatch')
        for key in ('validation_pair_ids_sha256', 'validation_labels_sha256', 'cost_config_sha256', 'patient_split_sha256'):
            if old[key] != new[key]:
                raise ValueError(f'8k/50k comparison input mismatch: {key}')
        old_cfg = yaml.safe_load((old_dir / 'resolved_config.yaml').read_text())
        new_cfg = yaml.safe_load((new_dir / 'resolved_config.yaml').read_text())
        if any(old_cfg[key] != new_cfg[key] for key in compared_settings):
            raise ValueError('8k/50k model or training settings differ')
        baseline_runs.append(old)
    result = analyze_learning_curve(experiment_root, report_root, expected_counts=(50000,))
    aggregate = result['aggregates']['50000']
    comparison = {}
    lines = ['# GREED 8k versus 50k', '', 'Three seeds; identical patient-disjoint validation labels and training settings.', '',
             '| Metric | 8k mean ± SD | 50k mean ± SD | Change |', '|---|---:|---:|---:|']
    for metric, label in (('validation_interval_loss', 'Interval loss'), ('validation_exact_mae', 'Exact MAE'),
                          ('validation_exact_rmse', 'Exact RMSE'), ('validation_exact_spearman', 'Spearman'),
                          ('validation_bounded_interval_coverage', 'Bounded coverage')):
        values = np.asarray([r['best_metrics'][metric] for r in baseline_runs], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'non-finite baseline metric: {metric}')
        old_mean, old_std = float(values.mean()), float(values.std(ddof=1))
        new_mean, new_std = aggregate[f'{metric}_mean'], aggregate[f'{metric}_std']
        comparison[metric] = {'baseline_mean': old_mean, 'baseline_std': old_std,
                              'scaled_mean': new_mean, 'scaled_std': new_std, 'change': new_mean - old_mean}
        lines.append(f'| {label} | {old_mean:.4f} ± {old_std:.4f} | {new_mean:.4f} ± {new_std:.4f} | {new_mean-old_mean:+.4f} |')
    result['comparison_8k'] = comparison
    write_json(report_root / 'greed_50k_comparison.json', result)
    _atomic_text(report_root / 'greed_50k_comparison.md', '\n'.join(lines) + '\n')
    return result
