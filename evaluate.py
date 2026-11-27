"""Evaluate a version-pinned W&B best model in a fresh process."""

import hydra
from omegaconf import DictConfig

from src.evaluation import run_evaluation


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
