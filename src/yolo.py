"""Ultralytics backend with resumable W&B checkpoints."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from src.data import prepare


def resolve_resume(reference, output):
    """Return last.pt and its run.json, either local or from a W&B artifact."""
    path = Path(reference).expanduser()
    if not path.is_file():
        artifact = reference.removeprefix("wandb://")
        if artifact.count("/") < 2 or ":" not in artifact:
            raise FileNotFoundError(f"Checkpoint not found: {reference}")
        import wandb

        directory = Path(wandb.Api().artifact(artifact, type="checkpoint").download(
            root=str(output / "download")
        ))
        path = directory / "last.pt"
    path = path.resolve()
    sidecar = path.parent / "run.json"
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("Resume needs last.pt and sibling run.json from a periodic checkpoint")
    return path, json.loads(sidecar.read_text(encoding="utf-8"))


def checkpoint_callback(run, info, every, online):
    """Called after save_model, before final evaluation strips optimizer state."""
    import wandb

    def save(trainer):
        epoch = trainer.epoch + 1
        if epoch % every and not trainer.stop:
            return
        artifact = wandb.Artifact(f"run_{run.id}_checkpoint", type="checkpoint",
                                  metadata={"epoch": epoch})
        artifact.add_file(str(trainer.last), name="last.pt")
        if trainer.best.exists():
            artifact.add_file(str(trainer.best), name="best.pt")
        sidecar = trainer.wdir / "run.json"
        sidecar.write_text(json.dumps(info, indent=2), encoding="utf-8")
        artifact.add_file(str(sidecar), name="run.json")
        logged = run.log_artifact(artifact, aliases=["latest"])
        if online:
            logged.wait()  # Do not announce a protected checkpoint until upload completes.
        print(f"Checkpoint epoch {epoch}: {artifact.name} ({'uploaded' if online else 'offline'})")

    return save


def train(cfg):
    if cfg.tracking.mode not in ("online", "offline"):
        raise ValueError("tracking.mode must be online or offline")
    every = int(cfg.tracking.checkpoint_every)
    if every < 1:
        raise ValueError("tracking.checkpoint_every must be positive")
    device = str(cfg.model.device)
    if device not in ("cpu", "0", "1"):
        raise ValueError("Use a single GPU (device=0 or 1), or cpu")
    if cfg.resume and cfg.tracking.mode != "online":
        raise ValueError("Resume uses the existing online W&B run; use tracking.mode=online")

    import wandb
    from ultralytics import YOLO, settings

    # Integration supplies epoch metrics, plots and the final best-model artifact.
    settings.update({"wandb": True, "mlflow": False, "comet": False,
                     "clearml": False, "dvc": False})
    output = Path(cfg.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    name = f"yolo26-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    save_dir = output / name
    checkpoint, previous = resolve_resume(str(cfg.resume), output / "downloads" / name) if cfg.resume else (None, None)
    data_yaml = prepare(cfg.data.root, cfg.data.prepared_dir, cfg.seed, cfg.data.val_fraction)
    preparation = json.loads((data_yaml.parent / "split.json").read_text(encoding="utf-8"))["preparation"]
    # Compare dataset/split identity, allowing its mount path to change between sessions.
    data_key = {k: preparation[k] for k in ("csv_sha256", "seed", "val_fraction")}
    if previous and previous["data"] != data_key:
        raise ValueError("Resume dataset or split differs from the checkpoint")
    model = YOLO(str(checkpoint) if checkpoint else cfg.model.weights)
    if checkpoint and (model.ckpt.get("epoch", -1) < 0 or model.ckpt.get("optimizer") is None):
        raise ValueError("This checkpoint has no optimizer/epoch state; use a periodic last.pt to resume")
    if checkpoint and not (checkpoint.parent / "best.pt").is_file():
        raise ValueError("Resume also needs sibling best.pt to preserve the previous best model")

    save_dir.mkdir(parents=True, exist_ok=False)
    with wandb.init(project=previous["project"] if previous else cfg.tracking.project,
                    entity=previous["entity"] if previous else cfg.tracking.entity,
                    id=previous["run_id"] if previous else None,
                    resume="must" if previous else None,
                    name=previous["name"] if previous else name,
                    mode=cfg.tracking.mode, dir=str(save_dir)) as run:
        for prefix in ("train", "val", "metrics", "lr"):
            run.define_metric(f"{prefix}/*", step_metric="epoch")
        info = {"run_id": run.id, "entity": run.entity, "project": run.project,
                "name": run.name, "data": data_key}

        def setup(trainer):
            (trainer.wdir / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            if checkpoint:
                shutil.copyfile(checkpoint.parent / "best.pt", trainer.best)

        model.add_callback("on_pretrain_routine_end", setup)
        model.add_callback("on_model_save", checkpoint_callback(run, info, every,
                                                               cfg.tracking.mode == "online"))
        args = dict(data=str(data_yaml), batch=cfg.model.batch_size, imgsz=cfg.model.imgsz,
                    device=device, workers=cfg.model.workers, save_dir=str(save_dir),
                    save_period=-1)
        if checkpoint:
            # Keep original optimizer, epoch budget and training recipe on resume.
            model.train(resume=True, **args)
        else:
            model.train(epochs=cfg.model.epochs, patience=cfg.model.patience,
                        amp=cfg.model.amp, seed=cfg.seed, deterministic=True,
                        fraction=cfg.model.fraction, cache=False, save=True,
                        project=str(output), name=name, **args)
        print(f"Best weights: {model.trainer.best}")
        print(f"W&B checkpoint: {run.entity}/{run.project}/run_{run.id}_checkpoint:latest")
