"""Optional public-test visualization; no labels and no AP evaluation."""

import hydra
from omegaconf import DictConfig

from src.evaluation import run_predictions


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_predictions(cfg)


if __name__ == "__main__":
    main()
