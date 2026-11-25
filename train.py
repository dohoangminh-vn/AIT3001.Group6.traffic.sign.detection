"""Single entrypoint: python train.py model=yolov8 model.batch_size=4."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.prepare_only:
        from src.data import prepare

        print(prepare(cfg.data.root, cfg.data.prepared_dir, cfg.seed, cfg.data.val_fraction))
    elif cfg.model.backend == "yolo":
        from src.yolo import train

        train(cfg)
    elif cfg.model.backend == "fasterrcnn":
        from src.fasterrcnn import train

        train(cfg)
    else:
        raise ValueError(f"Unsupported backend: {cfg.model.backend}")


if __name__ == "__main__":
    main()
