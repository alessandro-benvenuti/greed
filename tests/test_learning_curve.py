import json

import pytest

from vascular_ged.atomic import write_json
from vascular_ged.training import analyze_learning_curve


def _metadata(count: int, seed: int) -> dict[str, object]:
    scale = count / 1000
    return {
        "train_pair_count": count,
        "training_seed": seed,
        "best_metrics": {
            "epoch": 10,
            "validation_interval_loss": 4 / scale + seed % 3,
            "validation_mean_interval_violation": 2 / scale,
            "validation_bounded_interval_coverage": 0.5 + 0.01 * scale,
            "validation_bounded_mean_interval_violation": 1 / scale,
            "validation_exact_mae": 3 / scale,
            "validation_exact_rmse": 4 / scale,
            "validation_exact_spearman": 0.7 + 0.01 * scale,
            "validation_exact_kendall": 0.5 + 0.01 * scale,
        },
        "epochs_completed": 20,
        "total_job_runtime_seconds": 30 * scale,
        "peak_memory_bytes": 123456,
        "slurm_job_id": f"job-{count}-{seed}",
        "train_subset_pair_ids_sha256": f"train-{count}",
        "validation_pair_ids_sha256": "shared-validation",
        "git_commit": "deadbeef",
    }


def test_learning_curve_analysis_validates_and_aggregates_twelve_runs(tmp_path):
    experiment = tmp_path / "checkpoints"
    reports = tmp_path / "reports"
    for count in (1000, 2000, 4000, 8000):
        for seed in (161803, 271828, 314159):
            write_json(experiment / f"pairs_{count}/seed_{seed}/model_metadata.json", _metadata(count, seed))

    result = analyze_learning_curve(experiment, reports)

    assert result["runs"] == 12
    assert result["aggregates"]["1000"]["runs"] == 3
    assert result["aggregates"]["8000"]["validation_exact_mae_mean"] == pytest.approx(3 / 8)
    assert (reports / "greed_learning_curve_runs.csv").is_file()
    assert json.loads((reports / "greed_learning_curve_metrics.json").read_text())["runs"] == 12
    assert "| 8000 |" in (reports / "greed_learning_curve_report.md").read_text()


def test_learning_curve_analysis_rejects_incomplete_results(tmp_path):
    write_json(tmp_path / "pairs_1000/seed_1/model_metadata.json", _metadata(1000, 1))
    with pytest.raises(ValueError, match="pair counts"):
        analyze_learning_curve(tmp_path, tmp_path / "reports")
