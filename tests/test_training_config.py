from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]


def test_cpu_training_config_and_launchers_are_consistent() -> None:
    config = yaml.safe_load((ROOT / "configs/greed_training_config.yaml").read_text(encoding="utf-8"))
    preflight = (ROOT / "scripts/jean_zay/train_preflight.sbatch").read_text(encoding="utf-8")
    training = (ROOT / "scripts/jean_zay/train_cpu.sbatch").read_text(encoding="utf-8")

    assert config["model"] == {
        "layers": 8, "input_dim": 3, "hidden_dim": 64,
        "embedding_dim": 64, "pooling": "sum",
    }
    assert config["batch_size"] == 32
    assert config["workers"] == 4
    assert config["torch_threads"] == 4
    assert config["epochs"] == 100
    assert config["checkpoint_metric"] == "validation_interval_loss"
    assert config["augmentation"] == "none"
    assert config["resume"] is True
    assert "--preflight-only" in preflight
    assert "--cpus-per-task=8" in preflight
    assert "--cpus-per-task=8" in training
    assert "--mem" not in preflight
    assert "--mem" not in training
