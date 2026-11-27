"""GPU inference followed by CPU operating-point and small-object metrics."""

import csv
from collections import Counter

from src.analysis import analysis_stage, prepare_analysis, read_json, resolve_analysis, write_csv, write_json
from src.diagnostics import _load_dataset, _read_ground_truth, _read_predictions, match_detections


def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def collect_predictions(model, data_yaml, cfg, confidence, nms_iou, section):
    """Keep only CPU lists; never retain Results or CUDA tensors between images."""
    images, labels, names = _load_dataset(data_yaml)
    predictions = model.predict(
        source=[str(path) for path in images], stream=True, save=False, verbose=False,
        imgsz=int(cfg.model.imgsz), conf=float(confidence), iou=float(nms_iou),
        max_det=int(cfg.evaluation.inference.max_detections), batch=int(section.batch_size),
        device=section.device, half=bool(section.amp) and str(section.device) != "cpu",
        tile=section.tile, workers=0,
    )
    records = []
    for path, result in zip(images, predictions, strict=True):
        height, width = result.orig_shape
        records.append({"image": path.name,
                        "ground_truths": _read_ground_truth(labels / f"{path.stem}.txt", width, height),
                        "predictions": _read_predictions(result)})
    return records, names


def operating_metrics(records, names, confidence, match_iou, cutoffs):
    """Class-aware TP matching; a class mistake contributes one FP and one FN."""
    names = dict(sorted(names.items()))
    indices = {class_id: index for index, class_id in enumerate(names)}
    background = len(names)
    matrix = [[0] * (background + 1) for _ in range(background + 1)]
    counts = {key: Counter() for key in names}
    boundaries = [0] + sorted(map(int, cutoffs))
    size_rows = [{"area_min_px2": lower, "area_max_px2_exclusive": upper,
                  "gt_count": 0, "tp": 0} for lower, upper in zip(boundaries, boundaries[1:] + [None])]
    gt_matches = []
    for record in records:
        gt = record["ground_truths"]
        pred = [item for item in record["predictions"] if item["confidence"] >= confidence]
        matching = match_detections(gt, pred, match_iou)
        correct = {item["ground_truth_index"] for item in matching["matches"] if item["error_type"] == "true_positive"}
        for item in matching["matches"]:
            actual = gt[item["ground_truth_index"]]["class_id"]
            predicted = pred[item["prediction_index"]]["class_id"]
            matrix[indices[actual]][indices[predicted]] += 1
            if actual == predicted:
                counts[actual]["tp"] += 1
            else:
                counts[actual]["fn"] += 1
                counts[predicted]["fp"] += 1
        for index in matching["unmatched_ground_truths"]:
            actual = gt[index]["class_id"]
            counts[actual]["fn"] += 1
            matrix[indices[actual]][background] += 1
        for index in matching["unmatched_predictions"]:
            predicted = pred[index]["class_id"]
            counts[predicted]["fp"] += 1
            matrix[background][indices[predicted]] += 1
        for index, target in enumerate(gt):
            x1, y1, x2, y2 = target["box"]
            area = (x2 - x1) * (y2 - y1)
            detected = index in correct
            counts[target["class_id"]]["gt_count"] += 1
            gt_matches.append({"image": record["image"], "class_id": target["class_id"],
                               "area_px2": area, "detected": detected})
            for row in size_rows:
                if area >= row["area_min_px2"] and (row["area_max_px2_exclusive"] is None or area < row["area_max_px2_exclusive"]):
                    row["gt_count"] += 1
                    row["tp"] += int(detected)
                    break
    per_class = [{"class_id": key, "class_name": name, "gt_count": counts[key]["gt_count"],
                  **precision_recall(counts[key]["tp"], counts[key]["fp"], counts[key]["fn"])} for key, name in names.items()]
    for row in size_rows:
        row["recall"] = row["tp"] / row["gt_count"] if row["gt_count"] else None
    totals = precision_recall(*(sum(row[key] for row in per_class) for key in ("tp", "fp", "fn")))
    return totals, per_class, matrix, size_rows, gt_matches


def _save_native_metrics(result, names, output):
    ap = {int(class_id): (float(ap50), float(ap95)) for class_id, ap50, ap95 in
          zip(result.box.ap_class_index, result.box.ap50, result.box.ap, strict=True)}
    write_csv(output / "class_ap.csv", [
        {"class_id": key, "class_name": name, "ap50": ap.get(key, (None, None))[0],
         "ap50_95": ap.get(key, (None, None))[1]} for key, name in sorted(names.items())])
    classes = [names[int(key)] for key in result.box.ap_class_index]
    curve_rows = []
    for curve_name, (x, y, x_title, y_title) in zip(result.curves, result.curves_results, strict=True):
        for class_name, values in zip(classes, y, strict=True):
            curve_rows.extend({"curve": curve_name, "class_name": class_name, "x": float(a), "y": float(b),
                               "x_label": x_title, "y_label": y_title} for a, b in zip(x, values, strict=True))
    write_csv(output / "native_curves.csv", curve_rows, ["curve", "class_name", "x", "y", "x_label", "y_label"])
    return {"map50": float(result.box.map50), "map50_95": float(result.box.map)}


def run_evaluation(cfg):
    cfg, root, checkpoint, expected = resolve_analysis(cfg)
    with analysis_stage(root, "evaluation", cfg) as output:
        from src.small_object_metrics import (
            _load_validation_ground_truth, evaluate_coco_predictions,
            log_small_object_metrics, validate_evaluation_config,
        )
        validate_evaluation_config(cfg.evaluation.inference, cfg.evaluation.small_object_area_cutoffs_px2)
        if not 0 <= float(cfg.evaluation.conf) <= 1 or not 0 < float(cfg.evaluation.match_iou) <= 1:
            raise ValueError("Invalid evaluation.conf or evaluation.match_iou")
        data_yaml = prepare_analysis(cfg, expected)
        from ultralytics import YOLO, settings
        settings.update({"wandb": False})
        model = YOLO(str(checkpoint))
        # Native AP/curves are computed first; no training model exists in this process.
        native = model.val(
            data=str(data_yaml), imgsz=int(cfg.model.imgsz), batch=int(cfg.evaluation.batch_size),
            device=cfg.evaluation.device, half=bool(cfg.evaluation.amp) and str(cfg.evaluation.device) != "cpu",
            conf=0.001, iou=float(cfg.evaluation.inference.nms_iou), tile=cfg.evaluation.tile,
            max_det=int(cfg.evaluation.inference.max_detections), workers=int(cfg.evaluation.workers),
            plots=False, verbose=False, project=str(output), name="native", exist_ok=True,
        )
        _, _, names = _load_dataset(data_yaml)
        native_scores = _save_native_metrics(native, names, output)
        del native
        records, names = collect_predictions(model, data_yaml, cfg, min(0.001, float(cfg.evaluation.conf)),
                                             cfg.evaluation.inference.nms_iou, cfg.evaluation)
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        totals, per_class, matrix, size_rows, gt_matches = operating_metrics(
            records, names, float(cfg.evaluation.conf), float(cfg.evaluation.match_iou),
            cfg.evaluation.small_object_area_cutoffs_px2)
        write_csv(output / "operating_metrics.csv", per_class)
        write_csv(output / "size_recall.csv", size_rows)
        write_csv(output / "ground_truth_matches.csv", gt_matches, ["image", "class_id", "area_px2", "detected"])
        labels = list(names.values()) + ["background"]
        write_csv(output / "confusion_matrix.csv", [{"ground_truth": label, **dict(zip(labels, values))}
                                                    for label, values in zip(labels, matrix)], ["ground_truth"] + labels)
        curve = []
        for confidence in sorted(set([i / 20 for i in range(1, 21)] + [float(cfg.evaluation.conf)])):
            scores, *_ = operating_metrics(records, names, confidence, float(cfg.evaluation.match_iou),
                                           cfg.evaluation.small_object_area_cutoffs_px2)
            curve.append({"confidence": confidence, **scores})
        write_csv(output / "confidence_metrics.csv", curve)
        dataset, _ = _load_validation_ground_truth(cfg.data.root, cfg.data.prepared_dir)
        image_ids = {row["file_name"]: row["id"] for row in dataset["images"]}
        coco_predictions = []
        for record in records:
            for prediction in record["predictions"]:
                if prediction["confidence"] < 0.001:
                    continue
                x1, y1, x2, y2 = prediction["box"]
                coco_predictions.append({"image_id": image_ids[record["image"]], "category_id": prediction["class_id"],
                                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": prediction["confidence"]})
        coco50, coco95 = evaluate_coco_predictions(dataset, coco_predictions)
        log_small_object_metrics(dataset, coco_predictions, output / "small_object_metrics.csv", None,
                                 cfg.evaluation.small_object_area_cutoffs_px2)
        source = read_json(root / "run_manifest.json")
        summary = {"run_name": source["run_name"], "artifact_uri": source["artifact_uri"],
                   "source_run_id": source["run_id"], "source_wandb_url": source.get("wandb_url"),
                   "checkpoint_sha256": source["checkpoint_sha256"], "validation_split_sha256": source["validation_split_sha256"],
                   "validation_images": len(records), "ultralytics": native_scores,
                   "coco": {"ap50": coco50, "ap50_95": coco95}, "operating": totals,
                   "confidence": float(cfg.evaluation.conf), "ap_confidence": 0.001,
                   "match_iou": float(cfg.evaluation.match_iou), "nms_iou": float(cfg.evaluation.inference.nms_iou),
                   "imgsz": int(cfg.model.imgsz), "tile": cfg.evaluation.tile,
                   "max_detections": int(cfg.evaluation.inference.max_detections),
                   "matching": "Same-class IoU matching first; class mistakes count as FP plus FN; operating scores are micro averages."}
        write_json(output / "metrics.json", summary)
        import wandb
        with wandb.init(project=source["project"], entity=source["entity"], mode=cfg.tracking.mode,
                        job_type="evaluate", group=source["run_id"], name=f"{source['run_name']}-eval-{output.name}",
                        dir=str(output), config=summary) as run:
            if source["artifact_uri"]:
                run.use_artifact(source["artifact_uri"].removeprefix("wandb://"), type="model")
            values = {f"evaluation/{key}": value for key, value in native_scores.items()}
            values.update({f"operating/{key}": value for key, value in totals.items()})
            run.log(values)
            for row in csv.DictReader((output / "small_object_metrics.csv").read_text(encoding="utf-8-sig").splitlines()):
                for key in ("ap50", "ap50_95"):
                    if row[key]:
                        run.summary[f"small_eval/area_lt_{row['area_cutoff_px2_exclusive']}/{key}"] = float(row[key])
    return output


def run_predictions(cfg):
    cfg, root, checkpoint, expected = resolve_analysis(cfg, "diagnostic")
    with analysis_stage(root, "predictions", cfg) as output:
        prepare_analysis(cfg, expected)
        from src.data import resolve_data_root
        source = resolve_data_root(cfg.data.root) / "traffic_public_test" / "images"
        images = sorted(path for path in source.glob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not images:
            raise ValueError(f"No public-test images found at {source}")
        from ultralytics import YOLO, settings
        settings.update({"wandb": False})
        model = YOLO(str(checkpoint))
        results = model.predict(
            source=[str(path) for path in images], stream=True, save=False, verbose=False,
            imgsz=int(cfg.model.imgsz), conf=float(cfg.diagnostic.conf),
            iou=float(cfg.evaluation.inference.nms_iou), max_det=int(cfg.evaluation.inference.max_detections),
            device=cfg.diagnostic.device, batch=int(cfg.diagnostic.batch_size), tile=cfg.diagnostic.tile,
            half=bool(cfg.diagnostic.amp) and str(cfg.diagnostic.device) != "cpu",
        )
        for path, result in zip(images, results, strict=True):
            result.save(filename=str(output / f"{path.stem}.jpg"))
        write_json(output / "report.json", {"images": len(images), "source": str(source),
                                            "note": "Unlabeled public test: predictions only, no AP metrics."})
    return output
