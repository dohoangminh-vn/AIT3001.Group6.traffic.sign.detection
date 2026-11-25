"""Tune inference thresholds on the prepared validation split."""

import hydra
from omegaconf import DictConfig

from src.tuning import run_tuning


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_tuning(cfg)


if __name__ == "__main__":
    main()
