"""Validation-only inference threshold tuning for W&B YOLO runs."""

import csv
import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path

from src.data import prepare
from src.diagnostics import _load_dataset, _read_ground_truth, _read_predictions, match_detections, resolve_diagnostic_checkpoint

METRICS = {"map50_95", "map50", "f1", "precision", "recall"}


def _find_run(api, entity, project, name):
    matches = [run for run in api.runs(f"{entity}/{project}") if run.name == name]
    if not matches:
        raise ValueError(f"No W&B run named exactly {name!r} in {entity}/{project}")
    if len(matches) > 1:
        raise ValueError(f"Multiple W&B runs named {name!r}; run name must be unique")
    return matches[0]


def _validate_config(cfg):
    if not cfg.tuning.run_name:
        raise ValueError("Set tuning.run_name to the exact W&B run name")
    selected = list(cfg.tuning.thresholds)
    if not selected or any(item not in ("nms_iou", "conf") for item in selected):
        raise ValueError("tuning.thresholds must contain nms_iou and/or conf")
    for item in selected:
        section = cfg.tuning[item]
        values = [float(value) for value in section.values]
        if not values:
            raise ValueError(f"tuning.{item}.values cannot be empty")
        lower = 0.0 if item == "conf" else 0.0
        if any(value <= lower or value > 1 for value in values):
            raise ValueError(f"tuning.{item}.values must be greater than 0 and at most 1")
        if section.metric not in METRICS:
            raise ValueError(f"Unsupported tuning metric: {section.metric}")
    if not 0 < float(cfg.tuning.match_iou) <= 1:
        raise ValueError("tuning.match_iou must be greater than 0 and at most 1")


def _counts(model, image_paths, label_dir, cfg, confidence, nms_iou, names):
    predictions_stream = model.predict(
        source=[str(path) for path in image_paths], stream=True, save=False, verbose=False,
        conf=confidence, iou=nms_iou, max_det=int(cfg.evaluation.inference.max_detections),
        batch=int(cfg.tuning.batch_size), half=bool(cfg.tuning.amp) and str(cfg.tuning.device) != "cpu",
        imgsz=int(cfg.model.imgsz), tile=cfg.tuning.tile, device=cfg.tuning.device,
    )
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for image_path, result in zip(image_paths, predictions_stream, strict=True):
        from PIL import Image
        with Image.open(image_path) as image:
            width, height = image.size
        gt = _read_ground_truth(label_dir / f"{image_path.stem}.txt", width, height)
        pred = _read_predictions(result)
        matched = match_detections(gt, pred, float(cfg.tuning.match_iou))
        # A wrong-class overlap is one FP and one FN.
        totals["tp"] += matched["true_positives"]
        totals["fp"] += len(matched["unmatched_predictions"]) + matched["misclassifications"]
        totals["fn"] += len(matched["unmatched_ground_truths"]) + matched["misclassifications"]
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    return {**totals, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def _map(model, data_yaml, cfg, nms_iou, confidence=0.001):
    result = model.val(
        data=str(data_yaml), imgsz=int(cfg.model.imgsz), batch=int(cfg.tuning.batch_size),
        device=cfg.tuning.device, half=bool(cfg.tuning.amp) and str(cfg.tuning.device) != "cpu",
        conf=confidence, iou=nms_iou, max_det=int(cfg.evaluation.inference.max_detections),
        plots=False, verbose=False, workers=0,
    )
    return {"map50_95": float(result.box.map), "map50": float(result.box.map50)}


def _write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_tuning(cfg):
    _validate_config(cfg)
    import wandb
    api = wandb.Api()
    run = _find_run(api, cfg.tracking.entity, cfg.tracking.project, str(cfg.tuning.run_name))
    artifact_uri = f"{run.entity}/{run.project}/run_{run.id}_checkpoint:latest"
    data_yaml = prepare(cfg.data.root, cfg.data.prepared_dir, cfg.seed, cfg.data.val_fraction)
    images, label_dir, names = _load_dataset(data_yaml)
    split = json.loads((Path(data_yaml).parent / "split.json").read_text(encoding="utf-8"))
    split_ids = split.get("splits", {}).get("val", [])
    split_hash = hashlib.sha256(json.dumps(split_ids, separators=(",", ":")).encode()).hexdigest()
    output = Path(cfg.output_dir).resolve() / "tuning"
    slug = "".join(c if c.isalnum() or c in "._-" else "_" for c in run.name).strip("._-")
    output = output / f"{slug}-{datetime.now():%Y%m%d-%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="traffic-sign-tune-") as temporary:
        checkpoint = resolve_diagnostic_checkpoint(artifact_uri, temporary)
        from ultralytics import YOLO
        model = YOLO(str(checkpoint))
        if "nms_iou" in cfg.tuning.thresholds and bool(getattr(model.model, "end2end", False)):
            raise ValueError("This checkpoint is end-to-end and does not use NMS IoU")
        best = {"conf": float(cfg.tuning.fixed_conf), "nms_iou": float(cfg.tuning.fixed_nms_iou)}
        reports = {}
        order = [key for key in ("nms_iou", "conf") if key in cfg.tuning.thresholds]
        for key in order:
            section = cfg.tuning[key]
            rows = []
            for value in map(float, section.values):
                conf = (0.001 if key == "nms_iou" and section.metric.startswith("map")
                        else value if key == "conf" else best["conf"])
                iou = value if key == "nms_iou" else best["nms_iou"]
                scores = (_map(model, data_yaml, cfg, iou, conf) if section.metric.startswith("map")
                          else _counts(model, images, label_dir, cfg, conf, iou, names))
                rows.append({"threshold": value, "conf": conf, "nms_iou": iou,
                             **scores, "selected": False})
            winner = max(rows, key=lambda row: (
                row[section.metric], -row["threshold"] if key == "nms_iou" else row["threshold"]
            ))
            winner["selected"] = True
            best[key] = winner["threshold"]
            reports[key] = {"metric": section.metric, "value": winner["threshold"],
                            "score": winner[section.metric], "conf": winner["conf"],
                            "nms_iou": winner["nms_iou"]}
            _write_csv(output / f"{key}_results.csv", rows)
            if section.metric.startswith("map") and key == "conf":
                print("Note: selecting confidence by mAP can remove low-confidence detections and reduce recall.")
    result = {"run_name": run.name, "run_id": run.id, "checkpoint": artifact_uri,
              "validation_split_sha256": split_hash, "validation_images": len(images),
              "selected_thresholds": best, "selection": reports,
              "metric_matching_iou": float(cfg.tuning.match_iou),
              "class_names": {str(k): v for k, v in names.items()}}
    (output / "best_thresholds.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Selected thresholds: conf={best['conf']}, NMS IoU={best['nms_iou']}")
    print(f"Tuning results: {output}")
    print(f"Run diagnostics with: python diagnose.py diagnostic.checkpoint={artifact_uri} diagnostic.conf={best['conf']} evaluation.inference.nms_iou={best['nms_iou']}")
    return output
