"""Render an offline report and export a weights-free ZIP for one run."""

import hydra
from omegaconf import DictConfig

from src.reporting import export_reports


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not cfg.analysis.run_dir:
        raise ValueError("Set analysis.run_dir to the exact run directory to export")
    export_reports(cfg.analysis.run_dir, cfg.analysis.export_dir)


if __name__ == "__main__":
    main()
