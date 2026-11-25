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


def resolve_data_root(root):
    """Accept either dataset parent or the traffic_train directory itself."""
    root = Path(root).expanduser().resolve()
    return root.parent if root.name == "traffic_train" else root


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


def split_by_street(images, seed=42, val_fraction=0.2, restarts=1000):
    """Split complete streets using seeded local search for image/class balance."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if restarts < 1:
        raise ValueError("restarts must be positive")
    if len(images) < 2:
        raise ValueError("Need at least two images to split")

    groups = {}
    total_classes = [0] * 7
    for image_id, image in images.items():
        if "street_id" not in image:
            raise ValueError(f"Image {image_id} has no street_id")
        street_id = image["street_id"]
        group = groups.setdefault(street_id, {"images": [], "classes": [0] * 7})
        group["images"].append(image_id)
        for label in image["labels"]:
            class_id = int(label[0])
            group["classes"][class_id] += 1
            total_classes[class_id] += 1
    street_ids = sorted(groups)
    if len(street_ids) < 2:
        raise ValueError("Need at least two streets to create a grouped train/validation split")

    total_images = len(images)

    def score(image_count, class_counts):
        image_error = (image_count / total_images - val_fraction) ** 2
        class_error = sum(
            (class_counts[index] / total_classes[index] - val_fraction) ** 2
            for index in range(7) if total_classes[index]
        ) / sum(bool(count) for count in total_classes)
        return image_error + class_error

    rng = random.Random(seed)
    best_score, best_validation_streets = math.inf, None
    for _ in range(restarts):
        validation_streets = {street_id for street_id in street_ids if rng.random() < val_fraction}
        if not validation_streets:
            validation_streets.add(rng.choice(street_ids))
        elif len(validation_streets) == len(street_ids):
            validation_streets.remove(rng.choice(street_ids))

        image_count = sum(len(groups[street_id]["images"]) for street_id in validation_streets)
        class_counts = [sum(groups[street_id]["classes"][index]
                            for street_id in validation_streets) for index in range(7)]
        current_score = score(image_count, class_counts)

        while True:
            best_move = None

            def consider(candidate_streets, candidate_image_count, candidate_classes):
                nonlocal best_move
                candidate_score = score(candidate_image_count, candidate_classes)
                candidate_key = tuple(sorted(candidate_streets))
                if candidate_score < current_score - 1e-15 and (
                    best_move is None or (candidate_score, candidate_key) < (best_move[0], best_move[1])
                ):
                    best_move = (candidate_score, candidate_key, candidate_streets,
                                 candidate_image_count, candidate_classes)

            for street_id in street_ids:
                group = groups[street_id]
                if street_id in validation_streets:
                    if len(validation_streets) == 1:
                        continue
                    consider(validation_streets - {street_id},
                             image_count - len(group["images"]),
                             [class_counts[i] - group["classes"][i] for i in range(7)])
                else:
                    if len(validation_streets) == len(street_ids) - 1:
                        continue
                    consider(validation_streets | {street_id},
                             image_count + len(group["images"]),
                             [class_counts[i] + group["classes"][i] for i in range(7)])

            train_streets = [street_id for street_id in street_ids if street_id not in validation_streets]
            for remove_id in sorted(validation_streets):
                removed = groups[remove_id]
                for add_id in train_streets:
                    added = groups[add_id]
                    candidate = (validation_streets - {remove_id}) | {add_id}
                    consider(candidate,
                             image_count - len(removed["images"]) + len(added["images"]),
                             [class_counts[i] - removed["classes"][i] + added["classes"][i]
                              for i in range(7)])

            if best_move is None:
                break
            current_score, _, validation_streets, image_count, class_counts = best_move

        candidate_key = tuple(sorted(validation_streets))
        best_key = tuple(sorted(best_validation_streets)) if best_validation_streets is not None else None
        if best_score == math.inf or (current_score, candidate_key) < (best_score, best_key):
            best_score, best_validation_streets = current_score, set(validation_streets)

    validation_ids = sorted(image_id for street_id in best_validation_streets
                             for image_id in groups[street_id]["images"])
    train_streets = set(street_ids) - best_validation_streets
    train_ids = sorted(image_id for street_id in train_streets for image_id in groups[street_id]["images"])
    return train_ids, validation_ids, sorted(train_streets), sorted(best_validation_streets)


def read_annotations(root, return_stats=False):
    """Return deduplicated image records, class names, and optional audit counts."""
    root = resolve_data_root(root)
    images, names, box_ids = {}, {}, set()
    seen_annotations = set()
    duplicate_group_keys = set()
    geometry_classes = {}
    stats = {"raw_annotation_count": 0, "unique_annotation_count": 0,
             "duplicate_annotations_removed": 0, "duplicate_groups": 0,
             "cross_class_bbox_conflicts": 0}
    source = root / "traffic_train" / "annotation.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            try:
                image_id = int(row["image_id"])
                street_id = int(row["street_id"])
                category = int(row["category_id"])
                if category not in range(1, 8):
                    raise ValueError(f"Unexpected category: {category}")
                filename = row["file_name"]
                if Path(filename).name != filename or "/" in filename or "\\" in filename:
                    raise ValueError(f"Expected image basename: {filename}")
                record = (filename, int(row["width"]), int(row["height"]))
                if image_id not in images:
                    images[image_id] = {"file_name": filename, "width": record[1],
                                        "street_id": street_id,
                                        "height": record[2], "labels": []}
                image = images[image_id]
                if record != (image["file_name"], image["width"], image["height"]):
                    raise ValueError(f"Inconsistent metadata for image {image_id}")
                if image["street_id"] != street_id:
                    raise ValueError(f"Inconsistent street_id for image {image_id}")
                box_id = int(row["box_id"])
                if box_id in box_ids:
                    raise ValueError(f"Duplicate box_id: {box_id}")
                box_ids.add(box_id)
                bbox = tuple(map(float, ast.literal_eval(row["bbox"])))
                box = yolo_box(bbox, record[1], record[2])
                name = row["supercategory"]
                if category in names and names[category] != name:
                    raise ValueError(f"Inconsistent class name for {category}")
                names[category] = name
                stats["raw_annotation_count"] += 1
                annotation_key = (image_id, category, bbox)
                geometry_key = (image_id, bbox)
                previous_classes = geometry_classes.setdefault(geometry_key, set())
                previous_classes.add(category)
                if annotation_key in seen_annotations:
                    stats["duplicate_annotations_removed"] += 1
                    duplicate_group_keys.add(annotation_key)
                    continue
                seen_annotations.add(annotation_key)
                image["labels"].append((category - 1, *box))
                stats["unique_annotation_count"] += 1
            except (ValueError, KeyError, SyntaxError, TypeError) as exc:
                raise ValueError(f"{source}, row {line}: {exc}") from exc
    if set(names) != set(range(1, 8)):
        raise ValueError("Expected all seven categories in annotation.csv")
    filenames = [image["file_name"] for image in images.values()]
    if len(set(Path(name).stem for name in filenames)) != len(filenames):
        raise ValueError("Different image IDs must have unique filename stems")
    stats["duplicate_groups"] = len(duplicate_group_keys)
    stats["cross_class_bbox_conflicts"] = sum(len(categories) > 1 for categories in geometry_classes.values())
    result = (images, [names[k] for k in range(1, 8)])
    return (*result, stats) if return_stats else result


def prepare(root, prepared_dir, seed=42, val_fraction=0.2):
    root, output = resolve_data_root(root), Path(prepared_dir).expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError("prepared_dir must be outside the source dataset")
    source = root / "traffic_train" / "annotation.csv"
    signature = {"source": str(root), "csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                 "seed": seed, "val_fraction": val_fraction, "format": 3,
                 "dedup_policy": "exact_image_bbox_class_v1",
                 "split_policy": "street_grouped_local_search_v1"}
    images, names, annotation_stats = read_annotations(root, return_stats=True)
    train_ids, val_ids, train_streets, val_streets = split_by_street(images, seed, val_fraction)
    split = {"train": train_ids, "val": val_ids}
    def bbox_counts(image_ids):
        return [sum(1 for image_id in image_ids for label in images[image_id]["labels"]
                    if label[0] == class_id) for class_id in range(7)]

    split_stats = {
        "train_image_count": len(train_ids),
        "val_image_count": len(val_ids),
        "val_image_fraction": len(val_ids) / len(images),
        "train_street_count": len(train_streets),
        "val_street_count": len(val_streets),
        "train_bbox_counts_by_class": bbox_counts(train_ids),
        "val_bbox_counts_by_class": bbox_counts(val_ids),
    }
    manifest = {"preparation": signature, "splits": split,
                "street_ids": {"train": train_streets, "val": val_streets},
                "split_stats": split_stats, "annotation_stats": annotation_stats}
    manifest_path = output / "split.json"
    reused = manifest_path.exists()
    if reused and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Prepared data differs from current data/split/dedup policy; "
                         "choose a new data.prepared_dir to rebuild it")
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
    print(f"{'Reused' if reused else 'Prepared'} {len(train_ids)} train / {len(val_ids)} val images "
          f"({len(val_ids) / len(images):.1%} val across {len(val_streets)} streets); "
          f"removed {annotation_stats['duplicate_annotations_removed']} duplicate annotations, "
          f"found {annotation_stats['cross_class_bbox_conflicts']} same-box cross-class conflicts: {output}")
    return data_yaml
