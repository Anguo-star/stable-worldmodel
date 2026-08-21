from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]


def test_prejepa_uses_the_common_training_logger() -> None:
    source = (ROOT / "scripts/train/prejepa.py").read_text(encoding="utf-8")
    config = OmegaConf.load(ROOT / "scripts/train/config/prejepa.yaml")

    assert "from stable_worldmodel.loggers import build_training_logger" in source
    assert "training_logger = build_training_logger(cfg)" in source
    assert "logger=training_logger" in source
    assert config.logger_backend == "none"
    assert config.swanlab.enabled is False
    assert config.wandb.enabled is False
