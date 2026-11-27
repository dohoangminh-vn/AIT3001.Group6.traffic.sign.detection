"""Validation-only inference threshold tuning for W&B YOLO runs."""

import math
from src.analysis import analysis_stage, prepare_analysis, read_json, resolve_analysis, write_csv, write_json
from src.evaluation import collect_predictions, operating_metrics

METRICS = {"map50_95", "map50", "f1", "precision", "recall"}


def _find_run(api, entity, project, name):
    matches = [run for run in api.runs(f"{entity}/{project}") if run.name == name]
    if not matches:
        raise ValueError(f"No W&B run named exactly {name!r} in {entity}/{project}")
    if len(matches) > 1:
        raise ValueError(f"Multiple W&B runs named {name!r}; run name must be unique")
    return matches[0]


def _validate_config(cfg):
    selected = list(cfg.tuning.thresholds)
    if not selected or any(item not in ("nms_iou", "conf") for item in selected):
        raise ValueError("tuning.thresholds must contain nms_iou and/or conf")
    for item in selected:
        section = cfg.tuning[item]
        values = [float(value) for value in section["values"]]
        if not values:
            raise ValueError(f"tuning.{item}.values cannot be empty")
        if any(not math.isfinite(value) or value <= 0 or value > 1 for value in values):
            raise ValueError(f"tuning.{item}.values must be greater than 0 and at most 1")
        if section.metric not in METRICS:
            raise ValueError(f"Unsupported tuning metric: {section.metric}")
    if not 0 < float(cfg.tuning.match_iou) <= 1:
        raise ValueError("tuning.match_iou must be greater than 0 and at most 1")
    if not 0 <= float(cfg.tuning.fixed_conf) <= 1 or not 0 < float(cfg.tuning.fixed_nms_iou) <= 1:
        raise ValueError("Invalid tuning.fixed_conf or tuning.fixed_nms_iou")


def _map(model, data_yaml, cfg, nms_iou, output, confidence=0.001):
    result = model.val(
        data=str(data_yaml), imgsz=int(cfg.model.imgsz), batch=int(cfg.tuning.batch_size),
        device=cfg.tuning.device, half=bool(cfg.tuning.amp) and str(cfg.tuning.device) != "cpu",
        conf=confidence, iou=nms_iou, max_det=int(cfg.evaluation.inference.max_detections),
        tile=cfg.tuning.tile, plots=False, verbose=False, workers=0,
        project=str(output), name="native", exist_ok=True,
    )
    return {"map50_95": float(result.box.map), "map50": float(result.box.map50)}


def run_tuning(cfg):
    _validate_config(cfg)
    if cfg.tuning.run_name and not cfg.analysis.checkpoint and not cfg.analysis.run_dir:
        import wandb
        run = _find_run(wandb.Api(), cfg.tracking.entity, cfg.tracking.project, str(cfg.tuning.run_name))
        cfg.analysis.checkpoint = f"wandb://{run.entity}/{run.project}/run_{run.id}_model:best"
    cfg, root, checkpoint, expected = resolve_analysis(cfg, "tuning")
    if cfg.analysis.thresholds_file:
        raise ValueError("Do not supply analysis.thresholds_file when selecting new thresholds")
    with analysis_stage(root, "tuning", cfg) as output:
        data_yaml = prepare_analysis(cfg, expected)
        from ultralytics import YOLO, settings
        settings.update({"wandb": False})
        model = YOLO(str(checkpoint))
        if "nms_iou" in cfg.tuning.thresholds and bool(getattr(model.model, "end2end", False)):
            raise ValueError("This checkpoint is end-to-end and does not use NMS IoU")
        best = {"conf": float(cfg.tuning.fixed_conf), "nms_iou": float(cfg.tuning.fixed_nms_iou)}
        reports = {}
        order = [key for key in ("nms_iou", "conf") if key in cfg.tuning.thresholds]
        for key in order:
            section = cfg.tuning[key]
            rows = []
            cached = None
            for value in map(float, section["values"]):
                conf = (0.001 if key == "nms_iou" and section.metric.startswith("map")
                        else value if key == "conf" else best["conf"])
                iou = value if key == "nms_iou" else best["nms_iou"]
                if section.metric.startswith("map"):
                    scores = _map(model, data_yaml, cfg, iou, output, conf)
                else:
                    # Confidence filtering happens on CPU; NMS is run just once for a confidence sweep.
                    if cached is None or key == "nms_iou":
                        cached = collect_predictions(model, data_yaml, cfg,
                                                     min(map(float, section["values"])) if key == "conf" else conf,
                                                     iou, cfg.tuning)
                    records, names = cached
                    scores, *_ = operating_metrics(records, names, conf, float(cfg.tuning.match_iou),
                                                    cfg.evaluation.small_object_area_cutoffs_px2)
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
            write_csv(output / f"{key}_results.csv", rows)
            if section.metric.startswith("map") and key == "conf":
                print("Note: selecting confidence by mAP can remove low-confidence detections and reduce recall.")
        source = read_json(root / "run_manifest.json")
        result = {key: source[key] for key in ("run_name", "run_id", "artifact_uri", "checkpoint_sha256", "validation_split_sha256")}
        result.update({"selected_thresholds": best, "selection": reports,
                       "metric_matching_iou": float(cfg.tuning.match_iou), "imgsz": int(cfg.model.imgsz),
                       "tile": cfg.tuning.tile, "max_detections": int(cfg.evaluation.inference.max_detections)})
        write_json(output / "best_thresholds.json", result)
    print(f"Selected thresholds: conf={best['conf']}, NMS IoU={best['nms_iou']}")
    print(f"Tuning results: {output}")
    print(f"Threshold file: {output / 'best_thresholds.json'}")
    return output
