"""Evaluate a trained YOLO checkpoint on nested ground-truth area ranges."""

import ast
import csv
import json
import math
from pathlib import Path


DEFAULT_AREA_CUTOFFS = (128, 256, 512, 1024)


def validate_evaluation_config(inference_cfg, area_cutoffs):
    """Validate shared thresholds before building either backend's evaluator."""
    score = float(inference_cfg["score_threshold"])
    nms_iou = float(inference_cfg["nms_iou"])
    max_detections = int(inference_cfg["max_detections"])
    if not 0.0 <= score <= 1.0:
        raise ValueError("evaluation.inference.score_threshold must be in [0, 1]")
    if not 0.0 <= nms_iou <= 1.0:
        raise ValueError("evaluation.inference.nms_iou must be in [0, 1]")
    if max_detections < 1:
        raise ValueError("evaluation.inference.max_detections must be positive")
    cutoffs = tuple(int(value) for value in area_cutoffs)
    if not cutoffs or any(value <= 0 for value in cutoffs) or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("evaluation.small_object_area_cutoffs_px2 must contain unique positive values")
    return cutoffs


def _mean_valid(values):
    values = [float(value) for value in values if math.isfinite(float(value)) and float(value) >= 0]
    return sum(values) / len(values) if values else None


def _precision_metrics(precision, iou_thresholds):
    """Average COCO precision tensor over recall, classes, and valid IoUs."""
    thresholds = [float(value) for value in iou_thresholds]
    iou_index = min(range(len(thresholds)), key=lambda index: abs(thresholds[index] - 0.5))
    if hasattr(precision, "shape"):
        tensor_shape = precision.shape
    else:
        dimensions = []
        current = precision
        while isinstance(current, (list, tuple)):
            dimensions.append(len(current))
            current = current[0]
        tensor_shape = dimensions
    max_det_index = tensor_shape[-1] - 1
    area_index = 0
    ap50_values = []
    ap_values = []
    for threshold_index in range(tensor_shape[0]):
        for recall_index in range(tensor_shape[1]):
            for class_index in range(tensor_shape[2]):
                value = precision[threshold_index][recall_index][class_index][area_index][max_det_index]
                if float(value) >= 0:
                    ap_values.append(float(value))
                    if threshold_index == iou_index:
                        ap50_values.append(float(value))
    return _mean_valid(ap50_values), _mean_valid(ap_values)


def _load_validation_ground_truth(data_root, prepared_dir):
    from src.data import resolve_data_root
    data_root = resolve_data_root(data_root)
    prepared_dir = Path(prepared_dir).expanduser().resolve()
    source = data_root / "traffic_train" / "annotation.csv"
    split_path = prepared_dir / "split.json"
    if not source.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {source}")
    if not split_path.is_file():
        raise FileNotFoundError(f"Prepared split manifest not found: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    val_ids = {int(value) for value in split["splits"]["val"]}

    image_map = {}
    annotation_map = {}
    category_names = {}
    seen = set()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), 2):
            image_id = int(row["image_id"])
            if image_id not in val_ids:
                continue
            try:
                category_id = int(row["category_id"]) - 1
                bbox = tuple(map(float, ast.literal_eval(row["bbox"])))
                if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
                    raise ValueError("bbox must have four finite coordinates")
                x, y, width, height = bbox
                if x < 0 or y < 0 or width <= 0 or height <= 0:
                    raise ValueError("bbox coordinates and dimensions are invalid")
                image = {
                    "id": image_id,
                    "file_name": row["file_name"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                }
                if image_id in image_map and image_map[image_id] != image:
                    raise ValueError(f"Inconsistent image metadata for image_id={image_id}")
                image_map[image_id] = image
                category_names[category_id] = row["supercategory"]
                key = (image_id, category_id, bbox)
                if key in seen:
                    continue
                seen.add(key)
                annotation_map.setdefault(image_id, []).append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                })
            except (KeyError, TypeError, ValueError, SyntaxError) as error:
                raise ValueError(f"{source}, row {line_number}: {error}") from error

    missing = val_ids - set(image_map)
    if missing:
        raise ValueError(f"Validation split contains images without labeled annotations: {sorted(missing)[:10]}")
    images = [image_map[image_id] for image_id in sorted(image_map)]
    annotations = []
    for image_id in sorted(annotation_map):
        for annotation in annotation_map[image_id]:
            annotation["id"] = len(annotations) + 1
            annotations.append(annotation)
    categories = [{"id": category_id, "name": name}
                  for category_id, name in sorted(category_names.items())]
    dataset = {"images": images, "annotations": annotations, "categories": categories}
    return dataset, image_map


def _run_coco_area_eval(dataset, predictions, cutoff):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError("pycocotools is required for small-object evaluation") from error

    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {"images": dataset["images"], "categories": dataset["categories"], "annotations": []}
        coco_dt.createIndex()

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = [image["id"] for image in dataset["images"]]
    # COCOeval bounds are inclusive. nextafter makes this implement strict area < cutoff.
    evaluator.params.areaRng = [[0.0, math.nextafter(float(cutoff), -math.inf)]]
    evaluator.params.areaRngLbl = ["small"]
    evaluator.evaluate()
    evaluator.accumulate()
    return _precision_metrics(evaluator.eval["precision"], evaluator.params.iouThrs)


def evaluate_coco_predictions(dataset, predictions):
    """Return common COCO AP50 and AP50:95 for predictions in original coordinates."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError("pycocotools is required for validation evaluation") from error
    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {"images": dataset["images"], "categories": dataset["categories"], "annotations": []}
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = [image["id"] for image in dataset["images"]]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return float(evaluator.stats[1]), float(evaluator.stats[0])


def log_small_object_metrics(dataset, predictions, output_csv, run,
                             area_cutoffs=DEFAULT_AREA_CUTOFFS):
    """Write and log area-limited AP for predictions in original-image coordinates."""
    rows, log_values = [], {}
    cutoffs = tuple(int(value) for value in area_cutoffs)
    if not cutoffs or any(value <= 0 for value in cutoffs) or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("small-object area cutoffs must be unique positive pixel areas")
    for cutoff in sorted(cutoffs):
        eligible = [annotation for annotation in dataset["annotations"] if annotation["area"] < cutoff]
        eligible_images = {annotation["image_id"] for annotation in eligible}
        if not eligible:
            raise ValueError(f"No validation ground-truth boxes have area < {cutoff} px²")
        ap50, ap50_95 = _run_coco_area_eval(dataset, predictions, cutoff)
        if ap50 is None or ap50_95 is None:
            raise RuntimeError(f"COCO evaluator returned undefined AP for area < {cutoff} px²")
        suffix = f"area_lt_{cutoff}"
        rows.append({"area_cutoff_px2_exclusive": cutoff, "gt_bbox_count": len(eligible),
                     "gt_image_count": len(eligible_images), "ap50": ap50, "ap50_95": ap50_95})
        log_values[f"small_eval/{suffix}/AP50"] = ap50
        log_values[f"small_eval/{suffix}/AP50_95"] = ap50_95
        log_values[f"small_eval/{suffix}/gt_bbox_count"] = len(eligible)
        log_values[f"small_eval/{suffix}/gt_image_count"] = len(eligible_images)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "area_cutoff_px2_exclusive", "gt_bbox_count", "gt_image_count", "ap50", "ap50_95",
        ])
        writer.writeheader()
        writer.writerows(rows)
    run.log(log_values)
    return rows


def evaluate_small_objects(checkpoint, data_root, prepared_dir, output_csv, run,
                           imgsz=640, batch=4, device="0", workers=2,
                           inference_cfg=None, area_cutoffs=DEFAULT_AREA_CUTOFFS):
    """Run best-checkpoint validation and report metrics in original-image coordinates."""
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("ultralytics is required for small-object evaluation") from error

    from src.data import resolve_data_root
    data_root = resolve_data_root(data_root)
    dataset, image_map = _load_validation_ground_truth(data_root, prepared_dir)
    image_paths = [str(data_root / "traffic_train" / "images" / image["file_name"])
                   for image in dataset["images"]]
    model = YOLO(str(checkpoint))
    inference_cfg = inference_cfg or {}
    results = model.predict(source=image_paths, imgsz=imgsz, batch=batch, device=device,
                            workers=workers,
                            conf=float(inference_cfg.get("score_threshold", 0.001)),
                            iou=float(inference_cfg.get("nms_iou", 0.7)),
                            max_det=int(inference_cfg.get("max_detections", 300)),
                            stream=True, verbose=False)
    predictions = []
    for result in results:
        image_id = int(Path(result.path).stem)
        if image_id not in image_map:
            raise ValueError(f"Prediction returned unknown validation image: {result.path}")
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        coordinates = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        categories = boxes.cls.detach().cpu().tolist()
        metadata = image_map[image_id]
        for box, confidence, category_id in zip(coordinates, confidences, categories):
            x1, y1, x2, y2 = map(float, box)
            x1, x2 = max(0.0, min(x1, metadata["width"])), max(0.0, min(x2, metadata["width"]))
            y1, y2 = max(0.0, min(y1, metadata["height"])), max(0.0, min(y2, metadata["height"]))
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue
            predictions.append({
                "image_id": image_id,
                "category_id": int(category_id),
                "bbox": [x1, y1, width, height],
                "score": float(confidence),
            })

    ap50, ap50_95 = evaluate_coco_predictions(dataset, predictions)
    run.log({"val/AP50": ap50, "val/AP50_95": ap50_95})
    return log_small_object_metrics(dataset, predictions, output_csv, run, area_cutoffs)
