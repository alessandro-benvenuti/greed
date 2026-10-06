from pathlib import Path
import shutil

import pytest

from vascular_ged.audit import audit
from vascular_ged.metadata import GraphRecord


def _dataset(tmp_path: Path) -> tuple[Path, Path, list[GraphRecord]]:
    root = tmp_path / "dataset"
    graphs = root / "train/vtp"
    graphs.mkdir(parents=True)
    fixture = Path(__file__).parent / "fixtures/line_dhw.vtp"
    records = []
    for index in range(2):
        relative = f"train/vtp/graph_{index}.vtp"
        shutil.copyfile(fixture, root / relative)
        records.append(GraphRecord(f"graph_{index}", f"patient_{index}", "0", relative, 3, 2))
    patch_index = root / "patch_index.csv"
    patch_index.write_text("fixture\n", encoding="utf-8")
    return root, patch_index, records


def test_parallel_audit_matches_serial_and_streams_coordinates(tmp_path: Path):
    root, patch_index, records = _dataset(tmp_path)
    serial = audit(records, root, patch_index, workers=1)
    parallel = audit(records, root, patch_index, workers=2)
    assert parallel == serial
    assert serial["training_patch_count"] == 2
    assert serial["coordinate_min_dhw"] == [0.1, 0.2, 0.3]
    assert serial["coordinate_max_dhw"] == [0.7, 0.8, 0.9]
    assert serial["node_count_mismatches"] == 0
    assert serial["edge_count_mismatches"] == 0


def test_audit_rejects_paths_outside_training_vtp(tmp_path: Path):
    root, patch_index, records = _dataset(tmp_path)
    outside = root / "val/vtp/graph.vtp"
    outside.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parent / "fixtures/line_dhw.vtp", outside)
    record = GraphRecord("outside", "patient", "0", "val/vtp/graph.vtp", 3, 2)
    with pytest.raises(FileNotFoundError, match="train/vtp"):
        audit([record], root, patch_index)


def test_audit_records_normalized_edge_count_correction(tmp_path: Path):
    root, patch_index, _ = _dataset(tmp_path)
    path = root / "train/vtp/duplicate.vtp"
    text = (Path(__file__).parent / "fixtures/line_dhw.vtp").read_text(encoding="utf-8")
    text = text.replace("0 1 1 2", "0 1 1 0 1 2").replace("2 4</DataArray>", "2 4 6</DataArray>")
    path.write_text(text, encoding="utf-8")
    record = GraphRecord("duplicate", "patient", "0", "train/vtp/duplicate.vtp", 3, 3)
    report = audit([record], root, patch_index)
    assert report["duplicate_edges"] == 1
    assert report["edge_count_mismatches"] == 1
    assert report["normalized_edge_count_corrections"] == {"duplicate": 2}
