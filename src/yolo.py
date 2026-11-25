"""Ultralytics backend with resumable W&B checkpoints."""

import json
import hashlib
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from src.data import prepare
from src.run_naming import make_run_name


ULTRALYTICS_AUGMENTATION_KEYS = (
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "mosaic", "mixup", "cutmix",
    "copy_paste", "bgr", "multi_scale",
)


def configured_train_args(cfg):
    """Merge model-native augmentation knobs behind the shared enable switch."""
    train_args = OmegaConf.to_container(cfg.model.train_args, resolve=True)
    train_args = {key: value for key, value in train_args.items() if value is not None}
    if cfg.augmentation.enabled:
        model_augmentation = OmegaConf.select(cfg, "model.augmentation")
        if model_augmentation is not None:
            train_args.update(OmegaConf.to_container(model_augmentation, resolve=True))
    else:
        train_args.update({key: 0.0 for key in ULTRALYTICS_AUGMENTATION_KEYS})
        train_args["close_mosaic"] = 0
        # None lets Ultralytics install its default Blur/MedianBlur/ToGray/CLAHE.
        # An empty list disables this Albumentations branch entirely.
        train_args["augmentations"] = []
    return train_args


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

    def delete_previous_versions(artifact, current_version):
        """Delete older checkpoint versions after the replacement is uploaded."""
        api = wandb.Api()
        # W&B's artifact_versions expects the collection name, not the full
        # artifact identifier (which may include ``:vN``).
        collection = f"{run.entity}/{run.project}/run_{run.id}_checkpoint"
        for version in api.artifact_versions("checkpoint", collection):
            if version.version != current_version:
                version.delete()

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
            try:
                delete_previous_versions(logged, logged.version)
            except Exception as error:
                # Upload already succeeded; pruning is housekeeping and must
                # not abort the training run or invalidate the latest alias.
                print(f"Warning: checkpoint uploaded, but old versions could not be pruned: {error}")
        print(f"Checkpoint epoch {epoch}: {artifact.name} ({'uploaded' if online else 'offline'})")

    return save


def train(cfg):
    from src.small_object_metrics import validate_evaluation_config

    validate_evaluation_config(
        cfg.evaluation.inference, cfg.evaluation.small_object_area_cutoffs_px2
    )
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
    name = make_run_name(cfg.tracking.name, default_prefix=Path(cfg.model.weights).stem)
    save_dir = output / name
    checkpoint, previous = resolve_resume(str(cfg.resume), output / "downloads" / name) if cfg.resume else (None, None)
    data_yaml = prepare(cfg.data.root, cfg.data.prepared_dir, cfg.seed, cfg.data.val_fraction)
    prepared_dir = Path(cfg.data.prepared_dir).resolve()
    preparation = json.loads((prepared_dir / "split.json").read_text(encoding="utf-8"))["preparation"]
    # Compare dataset/split identity, allowing its mount path to change between sessions.
    data_key = {k: preparation[k] for k in
                ("csv_sha256", "seed", "val_fraction", "split_policy")}
    validation_ids = json.loads((prepared_dir / "split.json").read_text(encoding="utf-8"))["splits"]["val"]
    data_key["validation_split_sha256"] = hashlib.sha256(
        json.dumps(validation_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    preprocess_manifest = {"online": True, "library": "ultralytics", "size": int(cfg.model.imgsz),
                           "letterbox": "Ultralytics dataset LetterBox"}
    data_key["preprocess_signature"] = preprocess_manifest
    if previous and previous["data"] != data_key:
        raise ValueError("Resume dataset or split differs from the checkpoint")
    model = YOLO(str(checkpoint) if checkpoint else cfg.model.weights)
    if checkpoint and (model.ckpt.get("epoch", -1) < 0 or model.ckpt.get("optimizer") is None):
        raise ValueError("This checkpoint has no optimizer/epoch state; use a periodic last.pt to resume")
    if checkpoint and not (checkpoint.parent / "best.pt").is_file():
        raise ValueError("Resume also needs sibling best.pt to preserve the previous best model")

    save_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    (save_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
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
        run.config.update({
            "backend": cfg.model.backend,
            "weights": cfg.model.weights,
            "preprocess": preprocess_manifest,
            "augmentation_enabled": bool(cfg.augmentation.enabled),
            "resolved_config": resolved_config,
            "torch_version": __import__("torch").__version__,
            "torchvision_version": __import__("torchvision").__version__,
        }, allow_val_change=True)

        def setup(trainer):
            (trainer.wdir / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            if checkpoint:
                shutil.copyfile(checkpoint.parent / "best.pt", trainer.best)

        model.add_callback("on_pretrain_routine_end", setup)
        model.add_callback("on_model_save", checkpoint_callback(run, info, every,
                                                               cfg.tracking.mode == "online"))
        args = dict(data=str(data_yaml), batch=cfg.model.batch_size, imgsz=cfg.model.imgsz,
                    device=device, workers=cfg.model.workers, save_dir=str(save_dir),
                    save_period=-1,
                    conf=float(cfg.evaluation.inference.score_threshold),
                    iou=float(cfg.evaluation.inference.nms_iou),
                    max_det=int(cfg.evaluation.inference.max_detections))
        if checkpoint:
            # Keep original optimizer, epoch budget and training recipe on resume.
            model.train(resume=True, **args)
        else:
            train_args = configured_train_args(cfg)
            # Keep shared evaluator thresholds authoritative across backends.
            for key in ("conf", "iou", "max_det"):
                train_args.pop(key, None)
            scheduler_cfg = train_args.pop("scheduler", None)
            if scheduler_cfg:
                import torch

                scheduler_cfg = dict(scheduler_cfg)
                scheduler_name = scheduler_cfg.pop("name", "ReduceLROnPlateau")
                if scheduler_name != "ReduceLROnPlateau":
                    raise ValueError(f"Unsupported scheduler: {scheduler_name}")
                monitor = scheduler_cfg.pop("monitor", "metrics/mAP50-95(B)")
                scheduler_cfg.pop("mode", None)  # configured below for this metric
                scheduler = None

                def plateau_step(trainer):
                    nonlocal scheduler
                    if scheduler is None:
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            trainer.optimizer, mode="max", **scheduler_cfg
                        )
                    metric = trainer.metrics.get(monitor)
                    if metric is not None:
                        scheduler.step(float(metric))

                model.add_callback("on_fit_epoch_end", plateau_step)
            model.train(epochs=cfg.model.epochs, patience=cfg.model.patience,
                        amp=cfg.model.amp, seed=cfg.seed, deterministic=True,
                        fraction=cfg.model.fraction, cache=False, save=True,
                        project=str(output), name=name, **args, **train_args)
        try:
            from src.small_object_metrics import evaluate_small_objects

            small_metrics = evaluate_small_objects(
                checkpoint=model.trainer.best,
                data_root=cfg.data.root,
                prepared_dir=cfg.data.prepared_dir,
                output_csv=save_dir / "small_object_metrics.csv",
                run=run,
                imgsz=cfg.model.imgsz,
                batch=cfg.model.batch_size,
                device=device,
                workers=cfg.model.workers,
                inference_cfg=OmegaConf.to_container(cfg.evaluation.inference, resolve=True),
                area_cutoffs=cfg.evaluation.small_object_area_cutoffs_px2,
            )
            print(f"Small-object metrics: {save_dir / 'small_object_metrics.csv'} ({len(small_metrics)} area cutoffs)")
        except Exception as error:
            # Keep trained weights and standard metrics when optional area evaluation fails.
            print(f"Warning: small-object evaluation failed; no custom metrics were logged: {error}")
        print(f"Best weights: {model.trainer.best}")
        print(f"W&B checkpoint: {run.entity}/{run.project}/run_{run.id}_checkpoint:latest")
