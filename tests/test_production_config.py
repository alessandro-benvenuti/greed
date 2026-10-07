from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_production_config_matches_array_script() -> None:
    config = yaml.safe_load((ROOT / "configs/production_config.yaml").read_text(encoding="utf-8"))
    script = (ROOT / "scripts/jean_zay/production_array.sbatch").read_text(encoding="utf-8")

    assert config == {
        "method": "f2",
        "time_limit_seconds": 60,
        "threads_per_pair": 1,
        "chunk_size": 25,
        "array_concurrency": 8,
        "reuse_compatible_pilot_results": True,
        "coordinate_scale": 1.0,
        "cost_config_id": "vascular_3d_raw_v1",
        "similarity_lambda": 0.05,
        "bounded_labels": True,
    }
    assert "#SBATCH --array=0-399%8" in script
    assert "chunk_size=25" in script
    assert "--cost-config-id vascular_3d_raw_v1 --solver f2 --timeout 60" in script
    assert "--method f2 --timeout 60" in script
    assert '--reuse-dir "$GED_COMPUTATIONS_ROOT/results/pilot/shards"' in script
