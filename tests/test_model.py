import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
tg = pytest.importorskip("torch_geometric")

from vascular_ged.atomic import write_csv
from vascular_ged.training import VascularPairDataset, _evaluation_metrics, make_model


def test_siamese_symmetry_self_distance_and_reload(tmp_path):
    model = make_model(layers=2, hidden=8, embedding=8).eval()
    assert model.encoder.output[0].in_features == 8 * 3
    a = tg.data.Batch.from_data_list([tg.data.Data(x=torch.tensor([[0., 0., 0.], [1., 0., 0.]]), edge_index=torch.tensor([[0, 1], [1, 0]]))])
    b = tg.data.Batch.from_data_list([tg.data.Data(x=torch.tensor([[0., 0., 0.]]), edge_index=torch.empty((2, 0), dtype=torch.long))])
    assert torch.allclose(model(a, b), model(b, a), atol=1e-7)
    assert torch.allclose(model(a, a), torch.zeros(1), atol=1e-7)
    path = tmp_path / "model.pt"; torch.save(model.state_dict(), path)
    loaded = make_model(layers=2, hidden=8, embedding=8); loaded.load_state_dict(torch.load(path, weights_only=True)); loaded.eval()
    assert torch.allclose(model(a, b), loaded(a, b))


def _label_row(pair_id="pair", split="train"):
    return {
        "pair_id": pair_id, "greed_split": split,
        "sample_a": "a", "sample_b": "b", "patient_a": "1", "patient_b": "2",
        "graph_a_relative_path": "train/vtp/a.vtp", "graph_b_relative_path": "train/vtp/b.vtp",
        "node_count_a": "3", "node_count_b": "3", "edge_count_a": "2", "edge_count_b": "2",
        "lower_bound": "1", "upper_bound": "1", "is_exact": "True", "solver_status": "ok",
        "cost_config_id": "vascular_3d_raw_v1", "size_stratum": "small",
    }


def test_pair_dataset_preloads_unique_graphs_once(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    graph_dir = root / "train/vtp"
    graph_dir.mkdir(parents=True)
    fixture = Path(__file__).parent / "fixtures/line_dhw.vtp"
    shutil.copyfile(fixture, graph_dir / "a.vtp")
    shutil.copyfile(fixture, graph_dir / "b.vtp")
    row = _label_row()
    labels = tmp_path / "labels.csv"
    write_csv(labels, [row], list(row))

    from vascular_ged import training
    original = training.load_vtp
    loaded = []

    def counted(path):
        loaded.append(path)
        return original(path)

    monkeypatch.setattr(training, "load_vtp", counted)
    dataset = VascularPairDataset(labels, root, "vascular_3d_raw_v1", "train")
    dataset[0]; dataset[0]

    assert len(loaded) == 2
    assert set(dataset.graphs) == {"train/vtp/a.vtp", "train/vtp/b.vtp"}
    assert dataset[0][0].x.shape == (3, 3)
    assert dataset[0][0].edge_index.shape == (2, 4)


def test_pair_dataset_rejects_wrong_partition_and_same_patient(tmp_path):
    root = tmp_path / "dataset"
    graph_dir = root / "train/vtp"
    graph_dir.mkdir(parents=True)
    fixture = Path(__file__).parent / "fixtures/line_dhw.vtp"
    shutil.copyfile(fixture, graph_dir / "a.vtp")
    shutil.copyfile(fixture, graph_dir / "b.vtp")
    row = _label_row(split="validation")
    labels = tmp_path / "labels.csv"
    write_csv(labels, [row], list(row))
    with pytest.raises(ValueError, match="wrong GREED partition"):
        VascularPairDataset(labels, root, "vascular_3d_raw_v1", "train")

    row = _label_row(); row["patient_b"] = "1"
    write_csv(labels, [row], list(row))
    with pytest.raises(ValueError, match="same-patient"):
        VascularPairDataset(labels, root, "vascular_3d_raw_v1", "train")


def test_evaluation_metrics_use_exact_rows_for_regression_and_ranking():
    metrics = _evaluation_metrics(
        np.asarray([1.0, 2.5, 5.0]),
        np.asarray([1.0, 2.0, 4.0]),
        np.asarray([1.0, 3.0, 4.0]),
    )
    assert metrics["interval_loss"] == pytest.approx(1 / 3)
    assert metrics["interval_coverage"] == pytest.approx(2 / 3)
    assert metrics["mean_interval_violation"] == pytest.approx(1 / 3)
    assert metrics["exact_mae"] == pytest.approx(0.5)
    assert metrics["exact_rmse"] == pytest.approx(np.sqrt(0.5))
    assert metrics["exact_spearman"] == pytest.approx(1.0)
