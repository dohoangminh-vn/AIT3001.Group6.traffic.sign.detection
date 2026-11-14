"""Run post-training validation inference and write image-level error reports."""

import csv
import hashlib
import json
import math
import re
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from src.data import prepare


IMAGE_SUMMARY_FIELDS = (
    "image", "width", "height", "ground_truths", "predictions", "true_positives",
    "false_positives", "false_negatives", "misclassifications", "error_count", "overlay",
)
ERROR_FIELDS = (
    "image", "error_type", "ground_truth_class_id", "ground_truth_class",
    "predicted_class_id", "predicted_class", "confidence", "iou",
    "gt_x1", "gt_y1", "gt_x2", "gt_y2", "pred_x1", "pred_y1", "pred_x2", "pred_y2",
)

COLORS = {
    "true_positive": (26, 160, 84),
    "false_positive": (222, 48, 48),
    "false_negative": (238, 143, 24),
    "misclassification": (153, 67, 190),
}


def box_iou(first, second):
    """Calculate IoU for two pixel-coordinate xyxy boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(ground_truths, predictions, iou_threshold=0.5):
    """Match same-class boxes first, then identify overlapping class mistakes.

    Each detection is a mapping containing ``class_id`` and ``box`` (xyxy).
    A wrong-class overlap is one diagnostic event, while duplicate predictions
    remain false positives.
    """
    if not 0.0 < float(iou_threshold) <= 1.0:
        raise ValueError("match_iou must be greater than 0 and at most 1")

    used_ground_truths, used_predictions = set(), set()
    matches = []

    def consume_candidates(class_agnostic):
        candidates = []
        for ground_truth_index, ground_truth in enumerate(ground_truths):
            if ground_truth_index in used_ground_truths:
                continue
            for prediction_index, prediction in enumerate(predictions):
                if prediction_index in used_predictions:
                    continue
                same_class = ground_truth["class_id"] == prediction["class_id"]
                if (class_agnostic and same_class) or (not class_agnostic and not same_class):
                    continue
                overlap = box_iou(ground_truth["box"], prediction["box"])
                if overlap >= iou_threshold:
                    candidates.append((overlap, ground_truth_index, prediction_index))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        for overlap, ground_truth_index, prediction_index in candidates:
            if ground_truth_index in used_ground_truths or prediction_index in used_predictions:
                continue
            used_ground_truths.add(ground_truth_index)
            used_predictions.add(prediction_index)
            ground_truth = ground_truths[ground_truth_index]
            prediction = predictions[prediction_index]
            matches.append({
                "ground_truth_index": ground_truth_index,
                "prediction_index": prediction_index,
                "iou": overlap,
                "error_type": (
                    "true_positive" if ground_truth["class_id"] == prediction["class_id"]
                    else "misclassification"
                ),
            })

    # First make the conventional class-aware matches.
    consume_candidates(class_agnostic=False)
    # Of the remaining boxes, pair spatially overlapping different classes.
    consume_candidates(class_agnostic=True)
    matches.sort(key=lambda item: (item["ground_truth_index"], item["prediction_index"]))

    return {
        "matches": matches,
        "unmatched_ground_truths": [
            index for index in range(len(ground_truths)) if index not in used_ground_truths
        ],
        "unmatched_predictions": [
            index for index in range(len(predictions)) if index not in used_predictions
        ],
        "true_positives": sum(item["error_type"] == "true_positive" for item in matches),
        "misclassifications": sum(item["error_type"] == "misclassification" for item in matches),
    }


def resolve_diagnostic_checkpoint(reference, download_root):
    """Resolve a local best.pt or download best.pt from a W&B checkpoint artifact."""
    reference = str(reference).strip()
    local_path = Path(reference).expanduser()
    if local_path.is_file():
        return local_path.resolve()
    if local_path.is_absolute() or local_path.suffix.lower() == ".pt":
        raise FileNotFoundError(f"Local checkpoint not found: {reference}")

    artifact_reference = reference.removeprefix("wandb://")
    if artifact_reference.count("/") < 2 or ":" not in artifact_reference:
        raise FileNotFoundError(
            f"Checkpoint not found: {reference}. Pass a local best.pt or "
            "a W&B checkpoint artifact URI such as entity/project/run_x_checkpoint:latest."
        )

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb to download a checkpoint artifact from W&B") from exc

    try:
        directory = Path(wandb.Api().artifact(
            artifact_reference, type="checkpoint"
        ).download(root=str(download_root)))
    except Exception as exc:
        raise RuntimeError(
            f"Could not download W&B checkpoint artifact {artifact_reference!r}; "
            "check the URI, WANDB_API_KEY, and artifact permissions."
        ) from exc

    direct = directory / "best.pt"
    if direct.is_file():
        return direct.resolve()
    candidates = sorted(path.resolve() for path in directory.rglob("best.pt"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"W&B checkpoint artifact contains no best.pt: {artifact_reference}")
    raise ValueError(
        f"W&B checkpoint artifact contains multiple best.pt files: {artifact_reference}"
    )


def _to_list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _read_ground_truth(label_path, width, height):
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing validation label file: {label_path}")
    detections = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected five YOLO label values at {label_path}:{line_number}")
        try:
            class_id = int(fields[0])
            center_x, center_y, box_width, box_height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}") from exc
        values = (center_x, center_y, box_width, box_height)
        if class_id < 0 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < box_width <= 1
                and 0 < box_height <= 1):
            raise ValueError(f"Out-of-range YOLO label at {label_path}:{line_number}")
        detections.append({
            "class_id": class_id,
            "box": (
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            ),
        })
    return detections


def _read_predictions(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    coordinates = _to_list(boxes.xyxy)
    class_ids = _to_list(boxes.cls)
    confidences = _to_list(boxes.conf)
    return [
        {"class_id": int(class_id), "box": tuple(map(float, coordinates[index])),
         "confidence": float(confidences[index])}
        for index, class_id in enumerate(class_ids)
    ]


def _class_name(class_id, names):
    return names.get(class_id, f"class_{class_id}")


def _draw_box(draw, box, label, color, width, font, image_size, below=False):
    coordinates = tuple(round(value) for value in box)
    draw.rectangle(coordinates, outline=color, width=width)
    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    text_left = max(0, min(coordinates[0], image_size[0] - text_width - 4))
    if below:
        text_top = min(coordinates[3] + 4, max(0, image_size[1] - text_height - 2))
    else:
        text_top = max(0, coordinates[1] - text_height - 4)
    text_box = draw.textbbox((text_left, text_top), label, font=font)
    draw.rectangle(
        (text_box[0] - 2, text_box[1] - 1, text_box[2] + 2, text_box[3] + 1), fill=color
    )
    draw.text((text_left, text_top), label, fill=(255, 255, 255), font=font)


def _load_font(image_size):
    font_size = max(12, min(image_size) // 48)
    candidates = (
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _write_overlay(image_path, ground_truths, predictions, matching, names, destination):
    gt_status = {}
    prediction_status = {}
    for match in matching["matches"]:
        gt_status[match["ground_truth_index"]] = match["error_type"]
        prediction_status[match["prediction_index"]] = match["error_type"]
    for index in matching["unmatched_ground_truths"]:
        gt_status[index] = "false_negative"
    for index in matching["unmatched_predictions"]:
        prediction_status[index] = "false_positive"

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font(image.size)
    line_width = max(2, min(image.size) // 250)
    for index, ground_truth in enumerate(ground_truths):
        status = gt_status[index]
        label = f"GT {_class_name(ground_truth['class_id'], names)}"
        if status == "false_negative":
            label += " [missed]"
        elif status == "misclassification":
            label += " [wrong class]"
        _draw_box(
            draw, ground_truth["box"], label, COLORS[status], line_width, font, image.size,
            below=True,
        )
    for index, prediction in enumerate(predictions):
        status = prediction_status[index]
        label = f"PRED {_class_name(prediction['class_id'], names)} {prediction['confidence']:.2f}"
        if status == "false_positive":
            label += " [FP]"
        elif status == "misclassification":
            label += " [wrong class]"
        _draw_box(draw, prediction["box"], label, COLORS[status], line_width, font, image.size)
    image.save(destination, format="JPEG", quality=92)


def _error_record(image_name, error_type, ground_truth=None, prediction=None, iou=None, names=None):
    names = names or {}
    record = {
        "image": image_name,
        "error_type": error_type,
        "ground_truth_class_id": "",
        "ground_truth_class": "",
        "predicted_class_id": "",
        "predicted_class": "",
        "confidence": "",
        "iou": "" if iou is None else round(iou, 6),
        "gt_x1": "", "gt_y1": "", "gt_x2": "", "gt_y2": "",
        "pred_x1": "", "pred_y1": "", "pred_x2": "", "pred_y2": "",
    }
    if ground_truth is not None:
        record["ground_truth_class_id"] = ground_truth["class_id"]
        record["ground_truth_class"] = _class_name(ground_truth["class_id"], names)
        for key, value in zip(("gt_x1", "gt_y1", "gt_x2", "gt_y2"), ground_truth["box"]):
            record[key] = round(value, 2)
    if prediction is not None:
        record["predicted_class_id"] = prediction["class_id"]
        record["predicted_class"] = _class_name(prediction["class_id"], names)
        record["confidence"] = round(prediction["confidence"], 6)
        for key, value in zip(("pred_x1", "pred_y1", "pred_x2", "pred_y2"), prediction["box"]):
            record[key] = round(value, 2)
    return record


def _error_records(image_name, ground_truths, predictions, matching, names):
    records = []
    for match in matching["matches"]:
        if match["error_type"] == "misclassification":
            records.append(_error_record(
                image_name, "misclassification",
                ground_truths[match["ground_truth_index"]],
                predictions[match["prediction_index"]], match["iou"], names,
            ))
    for index in matching["unmatched_predictions"]:
        records.append(_error_record(
            image_name, "false_positive", prediction=predictions[index], names=names,
        ))
    for index in matching["unmatched_ground_truths"]:
        records.append(_error_record(
            image_name, "false_negative", ground_truth=ground_truths[index], names=names,
        ))
    return records


def _load_dataset(data_yaml):
    data_yaml = Path(data_yaml)
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(config.get("path") or data_yaml.parent)
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    validation = config["val"]
    if isinstance(validation, list):
        raise ValueError("Diagnostic mode expects one validation image directory")
    val_path = Path(validation)
    if not val_path.is_absolute():
        val_path = dataset_root / val_path
    if val_path.is_dir():
        image_paths = sorted(
            (path for path in val_path.iterdir()
             if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}),
            key=lambda path: path.name.casefold(),
        )
    elif val_path.is_file():
        image_paths = []
        for item in val_path.read_text(encoding="utf-8").splitlines():
            if item.strip():
                path = Path(item.strip())
                image_paths.append(path if path.is_absolute() else dataset_root / path)
    else:
        raise FileNotFoundError(f"Validation image path does not exist: {val_path}")
    if not image_paths:
        raise ValueError(f"No validation images found at {val_path}")

    names_config = config["names"]
    if isinstance(names_config, list):
        names = {index: str(value) for index, value in enumerate(names_config)}
    elif isinstance(names_config, dict):
        names = {int(key): str(value) for key, value in names_config.items()}
    else:
        raise ValueError("data.yaml names must be a list or class-ID mapping")
    label_dir = data_yaml.parent / "labels" / "val"
    return image_paths, label_dir, names


def _new_output_dir(output_root, checkpoint_reference):
    tail = checkpoint_reference.rsplit("/", 1)[-1].split(":", 1)[0]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", tail).strip("._-") or "model"
    slug = slug[-60:]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = output_root / "diagnostics" / f"{slug}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "images").mkdir()
    return output_dir


def run_diagnostics(cfg):
    """Run predictions over the prepared validation split and write a report."""
    reference = cfg.diagnostic.checkpoint
    if not reference:
        raise ValueError(
            "Set diagnostic.checkpoint to a local best.pt or a W&B checkpoint artifact URI."
        )
    confidence = float(cfg.diagnostic.conf)
    match_iou = float(cfg.diagnostic.match_iou)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("diagnostic.conf must be between 0 and 1")
    if not 0.0 < match_iou <= 1.0:
        raise ValueError("diagnostic.match_iou must be greater than 0 and at most 1")

    data_yaml = prepare(
        cfg.data.root, cfg.data.prepared_dir, cfg.seed, cfg.data.val_fraction
    )
    image_paths, label_dir, names = _load_dataset(data_yaml)
    split_manifest_path = Path(data_yaml).parent / "split.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    split_ids = split_manifest.get("splits", {}).get("val", [])
    split_hash = hashlib.sha256(
        json.dumps(split_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    preparation = split_manifest.get("preparation", {})

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the pinned dependencies from requirements.txt to run diagnostics") from exc

    output_root = Path(cfg.output_dir).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="traffic-sign-diagnose-") as temporary:
        checkpoint_path = resolve_diagnostic_checkpoint(reference, temporary)
        model = YOLO(str(checkpoint_path))
        output_dir = _new_output_dir(output_root, str(reference))
        image_summary_path = output_dir / "image_summary.csv"
        errors_path = output_dir / "errors.csv"
        totals = {
            "images": 0, "ground_truths": 0, "predictions": 0, "true_positives": 0,
            "false_positives": 0, "false_negatives": 0, "misclassifications": 0,
            "error_events": 0,
        }
        print(
            f"Diagnosing {len(image_paths)} validation images with confidence={confidence} "
            f"and match_iou={match_iou}."
        )

        with image_summary_path.open("w", newline="", encoding="utf-8-sig") as summary_stream, \
                errors_path.open("w", newline="", encoding="utf-8-sig") as errors_stream:
            summary_writer = csv.DictWriter(summary_stream, fieldnames=IMAGE_SUMMARY_FIELDS)
            errors_writer = csv.DictWriter(errors_stream, fieldnames=ERROR_FIELDS)
            summary_writer.writeheader()
            errors_writer.writeheader()

            prediction_stream = model.predict(
                source=[str(path) for path in image_paths],
                stream=True,
                save=False,
                verbose=False,
                conf=confidence,
                imgsz=int(cfg.model.imgsz),
                device=cfg.diagnostic.device,
            )
            processed = 0
            for image_path, result in zip(image_paths, prediction_stream, strict=True):
                with Image.open(image_path) as opened:
                    width, height = opened.size
                ground_truths = _read_ground_truth(
                    label_dir / f"{image_path.stem}.txt", width, height
                )
                predictions = _read_predictions(result)
                matching = match_detections(ground_truths, predictions, match_iou)
                records = _error_records(
                    image_path.name, ground_truths, predictions, matching, names
                )
                error_count = len(records)
                overlay = ""
                if error_count:
                    overlay_path = output_dir / "images" / f"{image_path.stem}.jpg"
                    _write_overlay(
                        image_path, ground_truths, predictions, matching, names, overlay_path
                    )
                    overlay = (Path("images") / overlay_path.name).as_posix()

                summary_writer.writerow({
                    "image": image_path.name,
                    "width": width,
                    "height": height,
                    "ground_truths": len(ground_truths),
                    "predictions": len(predictions),
                    "true_positives": matching["true_positives"],
                    "false_positives": len(matching["unmatched_predictions"]),
                    "false_negatives": len(matching["unmatched_ground_truths"]),
                    "misclassifications": matching["misclassifications"],
                    "error_count": error_count,
                    "overlay": overlay,
                })
                errors_writer.writerows(records)
                processed += 1
                totals["images"] += 1
                totals["ground_truths"] += len(ground_truths)
                totals["predictions"] += len(predictions)
                totals["true_positives"] += matching["true_positives"]
                totals["false_positives"] += len(matching["unmatched_predictions"])
                totals["false_negatives"] += len(matching["unmatched_ground_truths"])
                totals["misclassifications"] += matching["misclassifications"]
                totals["error_events"] += error_count
                if processed == 1 or processed % 100 == 0 or processed == len(image_paths):
                    print(f"Processed {processed}/{len(image_paths)} validation images.")
            if processed != len(image_paths):
                raise RuntimeError(
                    f"Model returned {processed} predictions for {len(image_paths)} validation images"
                )

        report = {
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "checkpoint": str(reference),
            "validation_images": len(image_paths),
            "confidence_threshold": confidence,
            "matching_iou_threshold": match_iou,
            "device": str(cfg.diagnostic.device),
            "imgsz": int(cfg.model.imgsz),
            "class_names": {str(key): value for key, value in names.items()},
            "data_csv_sha256": preparation.get("csv_sha256"),
            "split_seed": preparation.get("seed"),
            "validation_split_sha256": split_hash,
            "totals": totals,
            "error_count_definition": (
                "A class mismatch is counted as one diagnostic event. False positives and "
                "false negatives are unmatched boxes; these triage totals are not mAP."
            ),
            "output_dir": str(output_dir),
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Diagnostic report: {output_dir}")
    print(
        "Errors: "
        f"{totals['false_positives']} FP, {totals['false_negatives']} FN, "
        f"{totals['misclassifications']} class mismatches across {totals['images']} images."
    )
    return output_dir
