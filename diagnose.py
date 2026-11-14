"""Create visual error reports for a trained YOLO checkpoint."""

import hydra
from omegaconf import DictConfig

from src.diagnostics import run_diagnostics


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run_diagnostics(cfg)


if __name__ == "__main__":
    main()
