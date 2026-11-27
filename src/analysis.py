"""Run provenance and shared, version-pinned post-training context."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from omegaconf import OmegaConf


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_hash(split):
    return hashlib.sha256(json.dumps(split["splits"]["val"], separators=(",", ":")).encode()).hexdigest()


def task_overrides():
    from hydra.core.hydra_config import HydraConfig
    return list(HydraConfig.get().overrides.task) if HydraConfig.initialized() else []


def initialize_run(directory, cfg, run):
    """Save portable metadata before training, without environment secrets."""
    directory = Path(directory)
    (directory / "split").mkdir(exist_ok=True)
    for name in ("split.json", "train.txt", "val.txt"):
        shutil.copyfile(Path(cfg.data.prepared_dir) / name, directory / "split" / name)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), directory / "resolved_config.yaml")
    OmegaConf.save(OmegaConf.create(task_overrides()), directory / "overrides.yaml")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    versions = {}
    for name in ("torch", "torchvision", "ultralytics", "wandb", "hydra-core", "numpy", "pycocotools",
                 "matplotlib", "Pillow", "PyYAML", "albumentations"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    import torch
    write_json(directory / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": versions, "git_commit": commit, "git_dirty": dirty,
        "gpu": torch.cuda.get_device_name(int(cfg.model.device)) if str(cfg.model.device) != "cpu" and torch.cuda.is_available() else None,
    })
    write_json(directory / "run_manifest.json", {
        "run_name": directory.name, "wandb_run_name": run.name, "run_id": run.id,
        "entity": run.entity, "project": run.project, "wandb_url": run.url,
        "backend": cfg.model.backend, "git_commit": commit,
        "artifact_uri": None, "stages": {"training": {"status": "running"}},
    })


def publish_best(directory, checkpoint, run, online=True):
    """Publish weights and their context together; wait before advertising the URI."""
    import wandb
    directory = Path(directory)
    manifest = read_json(directory / "run_manifest.json")
    manifest["stages"]["training"] = {"status": "complete"}
    manifest["model_upload"] = "pending" if online else "offline"
    write_json(directory / "run_manifest.json", manifest)
    artifact = wandb.Artifact(f"run_{run.id}_model", type="model")
    artifact.add_file(str(checkpoint), name="best.pt")
    for name in ("run_manifest.json", "resolved_config.yaml", "overrides.yaml", "environment.json"):
        artifact.add_file(str(directory / name), name=name)
    artifact.add_dir(str(directory / "split"), name="split")
    if (directory / "results.csv").is_file():
        artifact.add_file(str(directory / "results.csv"), name="training/results.csv")
    logged = run.log_artifact(artifact, aliases=["best"])
    if online:
        logged.wait()
        manifest["artifact_uri"] = f"wandb://{run.entity}/{run.project}/{logged.name}"
        manifest["model_upload"] = "complete"
    write_json(directory / "run_manifest.json", manifest)
    return manifest


def _merge_source_config(cfg, saved):
    result = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for key in ("model", "data", "seed", "augmentation", "evaluation"):
        if key in saved:
            # Runtime evaluation controls retain the current defaults.
            if key == "evaluation":
                result.evaluation.inference = saved.evaluation.inference
                result.evaluation.small_object_area_cutoffs_px2 = saved.evaluation.small_object_area_cutoffs_px2
            else:
                result[key] = saved[key]
    if OmegaConf.select(saved, "model.train_args.tile") is not None:
        result.evaluation.tile = saved.model.train_args.tile
        result.diagnostic.tile = saved.model.train_args.tile
        result.tuning.tile = saved.model.train_args.tile
    overrides = [item.lstrip("+") for item in task_overrides()
                 if "=" in item and "." in item.split("=", 1)[0] and not item.startswith("~")]
    result = OmegaConf.merge(result, OmegaConf.from_dotlist(overrides))
    keys = {item.split("=", 1)[0] for item in overrides}
    for section in ("diagnostic", "tuning"):
        if f"{section}.tile" not in keys:
            result[section].tile = result.evaluation.tile
    return result


def resolve_analysis(cfg, section="evaluation"):
    """Download an immutable W&B model version and recover its training context."""
    reference = OmegaConf.select(cfg, "analysis.checkpoint")
    requested = OmegaConf.select(cfg, "analysis.run_dir")
    if not reference and requested:
        reference = read_json(Path(requested) / "run_manifest.json").get("artifact_uri")
    if not reference:
        raise ValueError("Set analysis.checkpoint to a W&B best-model artifact URI")
    reference = str(reference)
    local = Path(reference).expanduser()
    if local.is_file():
        root = Path(requested).resolve() if requested else local.resolve().parent.parent
        if not (root / "resolved_config.yaml").is_file():
            raise ValueError("Local analysis needs analysis.run_dir with resolved_config.yaml and split/split.json")
        source = root
        checkpoint = local.resolve()
        identity = {"artifact_uri": None, "checkpoint_sha256": sha256(checkpoint)}
    else:
        if local.suffix.lower() == ".pt" or local.is_absolute():
            raise FileNotFoundError(f"Local checkpoint not found: {reference}")
        import wandb
        uri = reference.removeprefix("wandb://")
        artifact = wandb.Api().artifact(uri, type="model")
        pinned = f"{artifact.entity}/{artifact.project}/{artifact.name}"
        cache = Path(cfg.analysis.cache_dir).resolve() / hashlib.sha256(pinned.encode()).hexdigest()[:20]
        source = Path(artifact.download(root=str(cache)))
        checkpoint = source / "best.pt"
        if not checkpoint.is_file() or not (source / "run_manifest.json").is_file():
            raise ValueError("This model artifact lacks best.pt or run metadata; use an artifact published by this workflow")
        identity = {"artifact_uri": f"wandb://{pinned}", "checkpoint_sha256": sha256(checkpoint)}
        source_manifest = read_json(source / "run_manifest.json")
        root = Path(requested).resolve() if requested else Path(cfg.output_dir).resolve() / source_manifest["run_name"]
        root.mkdir(parents=True, exist_ok=True)
        if (root / "run_manifest.json").exists():
            existing = read_json(root / "run_manifest.json")
            if existing["run_id"] != source_manifest["run_id"]:
                raise ValueError("analysis.run_dir belongs to another W&B run")
            if existing.get("checkpoint_sha256") and existing["checkpoint_sha256"] != identity["checkpoint_sha256"]:
                raise ValueError("This directory contains results for different weights; choose another analysis.run_dir")
        for name in ("resolved_config.yaml", "overrides.yaml", "environment.json", "split", "training"):
            src, dst = source / name, root / name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif src.is_file():
                shutil.copyfile(src, dst)
        if not (root / "run_manifest.json").exists():
            write_json(root / "run_manifest.json", source_manifest)
    saved = OmegaConf.load(source / "resolved_config.yaml")
    cfg = _merge_source_config(cfg, saved)
    if cfg.model.backend != "yolo":
        raise ValueError("Post-training analysis currently supports YOLO only")
    expected = read_json(source / "split" / "split.json")
    manifest = read_json(root / "run_manifest.json")
    prior = manifest.get("checkpoint_sha256")
    if prior and prior != identity["checkpoint_sha256"]:
        raise ValueError("This analysis directory already contains results for different weights; choose another analysis.run_dir")
    manifest.update(identity)
    manifest["validation_split_sha256"] = split_hash(expected)
    manifest["model_upload"] = "complete" if identity["artifact_uri"] else "local"
    write_json(root / "run_manifest.json", manifest)
    cfg.analysis.run_dir = str(root)
    if cfg.analysis.thresholds_file:
        thresholds = read_json(cfg.analysis.thresholds_file)
        for key in ("checkpoint_sha256", "validation_split_sha256"):
            if thresholds.get(key) != manifest[key]:
                raise ValueError(f"Threshold file differs in {key}")
        for key, value in (("imgsz", int(cfg.model.imgsz)), ("tile", cfg[section].tile),
                           ("max_detections", int(cfg.evaluation.inference.max_detections))):
            if thresholds.get(key) != value:
                raise ValueError(f"Threshold file differs in {key}")
        chosen = thresholds["selected_thresholds"]
        cfg.evaluation.conf = cfg.diagnostic.conf = float(chosen["conf"])
        cfg.evaluation.inference.nms_iou = float(chosen["nms_iou"])
        cfg.evaluation.match_iou = cfg.diagnostic.match_iou = float(thresholds["metric_matching_iou"])
    return cfg, root, checkpoint, expected


def prepare_analysis(cfg, expected):
    from src.data import prepare
    data_yaml = prepare(cfg.data.root, cfg.data.prepared_dir, cfg.seed, cfg.data.val_fraction)
    actual = read_json(Path(data_yaml).parent / "split.json")
    if actual["preparation"]["csv_sha256"] != expected["preparation"]["csv_sha256"] or split_hash(actual) != split_hash(expected):
        raise ValueError("Dataset CSV or validation split differs from the training run")
    return data_yaml


@contextmanager
def analysis_stage(root, name, cfg):
    variant = "tuned" if cfg.analysis.thresholds_file else "baseline"
    key = name if name == "tuning" else f"{name}/{variant}"
    path = root / key / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    path.mkdir(parents=True)
    OmegaConf.save(cfg, path / "config.yaml", resolve=True)
    manifest_path = root / "run_manifest.json"

    def record(status, error=None):
        manifest = read_json(manifest_path)
        manifest.setdefault("stages", {})[key] = {"status": status, "path": path.relative_to(root).as_posix()}
        if error:
            manifest["stages"][key]["error"] = str(error)
        write_json(manifest_path, manifest)

    record("running")
    try:
        yield path
    except Exception as error:
        record("failed", error)
        raise
    else:
        record("complete")
        print(f"{name}: {path}")
