from pathlib import Path

import pytest

from vascular_ged.atomic import write_csv
from vascular_ged.reporting import label_summary


def _row(pair_id: str, lower: float, upper: float) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "lower_bound": lower,
        "upper_bound": upper,
        "runtime_seconds": 1.5,
        "solver_status": "ok",
        "sampling_strategy": "size_matched",
        "size_stratum": "medium",
    }


def test_label_summary_counts_each_combined_row_once(tmp_path: Path) -> None:
    labels = tmp_path / "all.csv"
    rows = [_row("one", 1, 1), _row("two", 2, 3)]
    write_csv(labels, rows, list(rows[0]))
    output = tmp_path / "summary.md"

    label_summary([labels], output)

    summary = output.read_text(encoding="utf-8")
    assert "Rows: 2" in summary
    assert "Exact: 1; bounded: 1" in summary
    assert "Failures/timeouts retained: 0" in summary


def test_label_summary_rejects_overlapping_inputs(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    row = _row("duplicate", 1, 1)
    write_csv(labels, [row], list(row))

    with pytest.raises(ValueError, match="duplicate pair IDs"):
        label_summary([labels, labels], tmp_path / "summary.md")
