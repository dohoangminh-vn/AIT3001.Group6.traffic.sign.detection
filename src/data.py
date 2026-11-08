"""CSV -> fixed image split and YOLO labels; source images remain untouched."""

import ast
import csv
import hashlib
import json
import math
import random
import shutil
import warnings
from pathlib import Path

import yaml
from PIL import Image


def yolo_box(bbox, width, height):
    """Convert a valid COCO xywh box to normalized YOLO cxcywh."""
    if len(bbox) != 4:
        raise ValueError(f"Expected four bbox coordinates, got {bbox}")
    x, y, w, h = map(float, bbox)
    if not all(math.isfinite(v) for v in (x, y, w, h, width, height)):
        raise ValueError(f"Non-finite box or image size: {bbox}")
    if width <= 0 or height <= 0 or x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"Invalid bbox/image size: {bbox}, {width}x{height}")
    if x + w > width or y + h > height:
        raise ValueError(f"Box outside image: {bbox}, {width}x{height}")
    return ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)


def split_images(image_ids, seed=42, val_fraction=0.2):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    ids = sorted(set(image_ids))
    if len(ids) < 2:
        raise ValueError("Need at least two images to split")
    random.Random(seed).shuffle(ids)
    count = max(1, min(len(ids) - 1, round(len(ids) * val_fraction)))
    return sorted(ids[count:]), sorted(ids[:count])


def read_annotations(root):
    """Return image records and class names, usable by future backends too."""
    root = Path(root).resolve()
    images, names, box_ids = {}, {}, set()
    source = root / "traffic_train" / "annotation.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            try:
                image_id = int(row["image_id"])
                category = int(row["category_id"])
                if category not in range(1, 8):
                    raise ValueError(f"Unexpected category: {category}")
                filename = row["file_name"]
                if Path(filename).name != filename or "/" in filename or "\\" in filename:
                    raise ValueError(f"Expected image basename: {filename}")
                record = (filename, int(row["width"]), int(row["height"]))
                if image_id not in images:
                    images[image_id] = {"file_name": filename, "width": record[1],
                                        "height": record[2], "labels": []}
                image = images[image_id]
                if record != (image["file_name"], image["width"], image["height"]):
                    raise ValueError(f"Inconsistent metadata for image {image_id}")
                box_id = int(row["box_id"])
                if box_id in box_ids:
                    raise ValueError(f"Duplicate box_id: {box_id}")
                box_ids.add(box_id)
                box = yolo_box(ast.literal_eval(row["bbox"]), record[1], record[2])
                image["labels"].append((category - 1, *box))
                name = row["supercategory"]
                if category in names and names[category] != name:
                    raise ValueError(f"Inconsistent class name for {category}")
                names[category] = name
            except (ValueError, KeyError, SyntaxError, TypeError) as exc:
                raise ValueError(f"{source}, row {line}: {exc}") from exc
    if set(names) != set(range(1, 8)):
        raise ValueError("Expected all seven categories in annotation.csv")
    filenames = [image["file_name"] for image in images.values()]
    if len(set(Path(name).stem for name in filenames)) != len(filenames):
        raise ValueError("Different image IDs must have unique filename stems")
    return images, [names[k] for k in range(1, 8)]


def prepare(root, prepared_dir, seed=42, val_fraction=0.2):
    root, output = Path(root).resolve(), Path(prepared_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("prepared_dir must be outside the source dataset")
    source = root / "traffic_train" / "annotation.csv"
    signature = {"source": str(root), "csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                 "seed": seed, "val_fraction": val_fraction, "format": 1}
    images, names = read_annotations(root)
    train_ids, val_ids = split_images(images, seed, val_fraction)
    split = {"train": train_ids, "val": val_ids}
    manifest = {"preparation": signature, "splits": split}
    manifest_path = output / "split.json"
    reused = manifest_path.exists()
    if reused and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Prepared data differs from current data/split; choose another data.prepared_dir")
    if not reused and output.exists() and any(output.iterdir()):
        raise ValueError("Incomplete prepared directory; choose a new data.prepared_dir or remove it")

    copied = False
    for subset, ids in split.items():
        image_dir, label_dir = output / "images" / subset, output / "labels" / subset
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_id in ids:
            image = images[image_id]
            src = root / "traffic_train" / "images" / image["file_name"]
            if not src.is_file():
                raise FileNotFoundError(f"Missing source image: {src}")
            dst = image_dir / src.name
            label = label_dir / f"{src.stem}.txt"
            if reused:
                if not dst.is_file() or not label.is_file():
                    raise ValueError(f"Prepared file missing: {dst} or {label}; rebuild prepared directory")
                continue
            with Image.open(src) as opened:
                if opened.size != (image["width"], image["height"]):
                    raise ValueError(f"Image dimensions disagree with CSV: {src}")
            try:
                dst.symlink_to(src)
            except OSError:
                # Windows may require Developer Mode. Kaggle uses symlinks.
                if not copied:
                    warnings.warn("Symlinks unavailable; copying images into prepared directory instead")
                    copied = True
                shutil.copyfile(src, dst)
            label.write_text("".join(
                f"{row[0]} " + " ".join(f"{v:.10f}" for v in row[1:]) + "\n"
                for row in image["labels"]
            ), encoding="utf-8")

    # Source-relative lists can also be consumed by future model backends.
    for subset, ids in split.items():
        path = output / f"{subset}.txt"
        if not reused or not path.exists():
            path.write_text("".join(images[i]["file_name"] + "\n" for i in ids), encoding="utf-8")
    data_yaml = output / "data.yaml"
    if not reused or not data_yaml.exists():
        data_yaml.write_text(yaml.safe_dump({"path": str(output), "train": "images/train",
                                            "val": "images/val", "names": names},
                                           allow_unicode=True, sort_keys=False), encoding="utf-8")
    if not reused:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{'Reused' if reused else 'Prepared'} {len(train_ids)} train / {len(val_ids)} val images: {output}")
    return data_yaml
