"""Generate dataset EDA tables for ZA Traffic 2020 annotations."""

import argparse
import ast
import csv
import math
from collections import defaultdict
from pathlib import Path


SIZE_BINS = (
    ("small", 0.0, 32**2),
    ("medium", 32**2, 96**2),
    ("large", 96**2, math.inf),
)
OVERLAP_THRESHOLDS = (0.0, 0.1, 0.5, 0.75)


def _quantile(values, probability):
    """Return a linearly interpolated quantile for a non-empty sequence."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _iou(first, second):
    """Compute IoU for xyxy boxes."""
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def analyze(root, output_dir):
    """Read labeled training annotations and write CSV tables plus summary."""
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    source = root / "traffic_train" / "annotation.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {source}")

    raw_images = {}
    classes = {}
    raw_annotations = []
    duplicate_groups = defaultdict(list)
    geometry_classes = defaultdict(set)
    geometry_annotations = defaultdict(list)
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), 2):
            try:
                image_id = row["image_id"]
                street_id = row["street_id"]
                category_id = int(row["category_id"])
                class_name = row["supercategory"]
                width, height = int(row["width"]), int(row["height"])
                bbox = ast.literal_eval(row["bbox"])
                if len(bbox) != 4:
                    raise ValueError("bbox must have four values")
                x, y, box_width, box_height = map(float, bbox)
                if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
                    raise ValueError("bbox coordinates must be finite")
                if width <= 0 or height <= 0 or box_width <= 0 or box_height <= 0:
                    raise ValueError("image and bbox dimensions must be positive")
            except (KeyError, TypeError, ValueError, SyntaxError) as error:
                raise ValueError(f"{source}, row {line_number}: {error}") from error

            if category_id in classes and classes[category_id] != class_name:
                raise ValueError(f"Inconsistent class name for category {category_id}")
            classes[category_id] = class_name
            image = raw_images.setdefault(image_id, {
                "street_ids": set(), "width": width, "height": height,
            })
            if image["width"] != width or image["height"] != height:
                raise ValueError(f"Inconsistent image dimensions for image_id {image_id}")
            image["street_ids"].add(street_id)
            record = {
                "image_id": image_id,
                "street_id": street_id,
                "category_id": category_id,
                "class_name": class_name,
                "width": width,
                "height": height,
                "bbox": (x, y, box_width, box_height),
                "box": (x, y, x + box_width, y + box_height),
                "box_id": row.get("box_id", ""),
                "source_row": line_number,
            }
            raw_annotations.append(record)
            duplicate_key = (image_id, category_id, record["bbox"])
            duplicate_groups[duplicate_key].append(record)
            geometry_key = (image_id, record["bbox"])
            geometry_classes[geometry_key].add(category_id)
            geometry_annotations[geometry_key].append(record)

    if not raw_images:
        raise ValueError(f"No annotation rows found in {source}")
    street_conflicts = {image_id: sorted(image["street_ids"]) for image_id, image in raw_images.items()
                        if len(image["street_ids"]) != 1}
    if street_conflicts:
        sample = "; ".join(f"{image_id}: {streets}" for image_id, streets in list(street_conflicts.items())[:10])
        raise ValueError(f"Images must map to exactly one street_id; conflicts: {sample}")

    duplicate_rows = [records for records in duplicate_groups.values() if len(records) > 1]
    duplicate_annotations_removed = sum(len(records) - 1 for records in duplicate_rows)
    cross_class_geometry = [records for key, records in geometry_annotations.items()
                            if len(geometry_classes[key]) > 1]
    unique_annotations = [records[0] for records in duplicate_groups.values()]

    images = {}
    street_images = defaultdict(set)
    street_boxes = defaultdict(int)
    class_boxes = defaultdict(int)
    class_images = defaultdict(set)
    size_boxes = defaultdict(int)
    size_images = defaultdict(set)
    class_size_boxes = defaultdict(int)
    areas = []
    for image_id, metadata in raw_images.items():
        images[image_id] = {
            "image_id": image_id,
            "street_ids": metadata["street_ids"],
            "width": metadata["width"],
            "height": metadata["height"],
            "boxes": 0,
            "classes": set(),
            "areas": [],
            "sizes": defaultdict(int),
            "annotations": [],
        }
    for record in unique_annotations:
        image_id = record["image_id"]
        category_id = record["category_id"]
        street_id = record["street_id"]
        _, _, box_width, box_height = record["bbox"]
        image = images[image_id]
        image["boxes"] += 1
        image["classes"].add(category_id)
        image["annotations"].append({
            "category_id": category_id,
            "class_name": record["class_name"],
            "box": record["box"],
        })

        area = box_width * box_height
        areas.append(area)
        image["areas"].append(area)
        class_boxes[category_id] += 1
        class_images[category_id].add(image_id)
        street_images[street_id].add(image_id)
        street_boxes[street_id] += 1

        size_name = next(name for name, lower, upper in SIZE_BINS if lower <= area < upper)
        image["sizes"][size_name] += 1
        size_boxes[size_name] += 1
        size_images[size_name].add(image_id)
        class_size_boxes[(category_id, size_name)] += 1

    overlap_pairs = []
    overlapping_images = {threshold: set() for threshold in OVERLAP_THRESHOLDS}
    overlapping_pair_counts = {threshold: 0 for threshold in OVERLAP_THRESHOLDS}
    total_image_pairs = 0
    for image_id, image in images.items():
        annotations = image["annotations"]
        total_image_pairs += len(annotations) * (len(annotations) - 1) // 2
        for first_index in range(len(annotations)):
            first = annotations[first_index]
            for second_index in range(first_index + 1, len(annotations)):
                second = annotations[second_index]
                iou = _iou(first["box"], second["box"])
                for threshold in OVERLAP_THRESHOLDS:
                    if iou > threshold if threshold == 0 else iou >= threshold:
                        overlapping_pair_counts[threshold] += 1
                        overlapping_images[threshold].add(image_id)
                if iou > 0:
                    overlap_pairs.append({
                        "image_id": image_id,
                        "street_id": next(iter(image["street_ids"])),
                        "first_category_id": first["category_id"] - 1,
                        "first_class_name": first["class_name"],
                        "second_category_id": second["category_id"] - 1,
                        "second_class_name": second["class_name"],
                        "iou": iou,
                    })
    overlap_pairs.sort(key=lambda row: (-row["iou"], int(row["image_id"]),
                                        row["first_category_id"], row["second_category_id"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    total_boxes = len(areas)
    image_rows = []
    for image_id, image in sorted(images.items(), key=lambda item: (int(item[0]), item[0])):
        image_rows.append({
            "image_id": image_id,
            "street_id": next(iter(image["street_ids"])),
            "width": image["width"],
            "height": image["height"],
            "bbox_count": image["boxes"],
            "class_count": len(image["classes"]),
            "bbox_area_min_px2": min(image["areas"]),
            "bbox_area_median_px2": _quantile(image["areas"], 0.5),
            **{f"{name}_bbox_count": image["sizes"][name] for name, _, _ in SIZE_BINS},
        })
    _write_csv(output_dir / "image_stats.csv", [
        "image_id", "street_id", "width", "height", "bbox_count", "class_count",
        "bbox_area_min_px2", "bbox_area_median_px2", "small_bbox_count",
        "medium_bbox_count", "large_bbox_count",
    ], image_rows)

    class_rows = [{
        "category_id": category_id - 1,
        "source_category_id": category_id,
        "class_name": class_name,
        "bbox_count": class_boxes[category_id],
        "image_count": len(class_images[category_id]),
        "bbox_fraction": class_boxes[category_id] / total_boxes,
    } for category_id, class_name in sorted(classes.items())]
    _write_csv(output_dir / "class_counts.csv", [
        "category_id", "source_category_id", "class_name", "bbox_count", "image_count", "bbox_fraction",
    ], class_rows)

    street_rows = [{
        "street_id": street_id,
        "image_count": len(street_images[street_id]),
        "bbox_count": street_boxes[street_id],
        "image_fraction": len(street_images[street_id]) / len(images),
    } for street_id in sorted(street_images, key=lambda value: (-len(street_images[value]), value))]
    _write_csv(output_dir / "street_counts.csv", [
        "street_id", "image_count", "bbox_count", "image_fraction",
    ], street_rows)

    size_rows = [{
        "size": name,
        "area_min_px2_inclusive": lower,
        "area_max_px2_exclusive": "inf" if math.isinf(upper) else upper,
        "bbox_count": size_boxes[name],
        "bbox_fraction": size_boxes[name] / total_boxes,
        "image_count": len(size_images[name]),
        "image_fraction": len(size_images[name]) / len(images),
    } for name, lower, upper in SIZE_BINS]
    _write_csv(output_dir / "box_size_distribution.csv", [
        "size", "area_min_px2_inclusive", "area_max_px2_exclusive", "bbox_count",
        "bbox_fraction", "image_count", "image_fraction",
    ], size_rows)

    class_size_rows = [{
        "category_id": category_id - 1,
        "class_name": classes[category_id],
        "size": size_name,
        "bbox_count": class_size_boxes[(category_id, size_name)],
        "class_bbox_fraction": class_size_boxes[(category_id, size_name)] / class_boxes[category_id],
    } for category_id in sorted(classes) for size_name, _, _ in SIZE_BINS]
    _write_csv(output_dir / "class_size_counts.csv", [
        "category_id", "class_name", "size", "bbox_count", "class_bbox_fraction",
    ], class_size_rows)

    class_size_lookup = {(row["category_id"], row["size"]): row for row in class_size_rows}
    image_bbox_counts = [image["boxes"] for image in images.values()]
    image_bbox_histogram = defaultdict(int)
    for count in image_bbox_counts:
        image_bbox_histogram[count] += 1
    overlap_class_pair_counts = defaultdict(int)
    for pair in overlap_pairs:
        first_id, second_id = sorted((pair["first_category_id"], pair["second_category_id"]))
        overlap_class_pair_counts[(first_id, second_id)] += 1
    overlap_ious = [pair["iou"] for pair in overlap_pairs]

    overlap_threshold_rows = [{
        "threshold": threshold,
        "comparison": ">" if threshold == 0 else ">=",
        "total_pairs_same_image": total_image_pairs,
        "overlapping_pairs": overlapping_pair_counts[threshold],
        "pair_fraction": overlapping_pair_counts[threshold] / total_image_pairs if total_image_pairs else 0.0,
        "images_with_overlap": len(overlapping_images[threshold]),
        "image_fraction": len(overlapping_images[threshold]) / len(images),
    } for threshold in OVERLAP_THRESHOLDS]
    _write_csv(output_dir / "overlap_thresholds.csv", [
        "threshold", "comparison", "total_pairs_same_image", "overlapping_pairs", "pair_fraction",
        "images_with_overlap", "image_fraction",
    ], overlap_threshold_rows)
    _write_csv(output_dir / "overlap_pairs.csv", [
        "image_id", "street_id", "first_category_id", "first_class_name",
        "second_category_id", "second_class_name", "iou",
    ], overlap_pairs)
    duplicate_audit_rows = [{
        "image_id": records[0]["image_id"],
        "street_id": records[0]["street_id"],
        "source_category_id": records[0]["category_id"],
        "class_name": records[0]["class_name"],
        "bbox_xywh": str(records[0]["bbox"]),
        "annotation_count": len(records),
        "duplicates_removed": len(records) - 1,
        "box_ids": ";".join(str(record["box_id"]) for record in records),
        "source_rows": ";".join(str(record["source_row"]) for record in records),
    } for records in duplicate_rows]
    _write_csv(output_dir / "duplicate_annotations.csv", [
        "image_id", "street_id", "source_category_id", "class_name", "bbox_xywh",
        "annotation_count", "duplicates_removed", "box_ids", "source_rows",
    ], duplicate_audit_rows)
    cross_class_rows = [{
        "image_id": records[0]["image_id"],
        "street_id": records[0]["street_id"],
        "bbox_xywh": str(records[0]["bbox"]),
        "source_category_ids": ";".join(str(value) for value in sorted({record["category_id"] for record in records})),
        "class_names": ";".join(classes[value] for value in sorted({record["category_id"] for record in records})),
        "annotation_count": len(records),
        "box_ids": ";".join(str(record["box_id"]) for record in records),
        "source_rows": ";".join(str(record["source_row"]) for record in records),
    } for records in cross_class_geometry]
    _write_csv(output_dir / "cross_class_bbox_conflicts.csv", [
        "image_id", "street_id", "bbox_xywh", "source_category_ids", "class_names",
        "annotation_count", "box_ids", "source_rows",
    ], cross_class_rows)

    lines = [
        "# ZA Traffic 2020 — EDA summary", "",
        f"- Annotation source: `{source}`",
        f"- Labeled images: {len(images):,}",
        f"- Raw annotation rows: {len(raw_annotations):,}",
        f"- Exact duplicate rows removed: {duplicate_annotations_removed:,} across {len(duplicate_rows):,} duplicate groups.",
        f"- Unique bbox annotations used for every EDA table: {total_boxes:,}",
        f"- Same-image, same-coordinate boxes with different classes (retained and flagged): {len(cross_class_rows):,}.",
        f"- Classes: {len(classes)}",
        f"- Streets: {len(street_images)}",
        "- Duplicate key: same image ID + exact bbox `[x, y, w, h]` coordinates + same class. The source CSV is not modified.",
        "- Area is computed from original pixel bbox width × height; COCO bins use small < 1,024 px², medium 1,024–< 9,216 px², and large ≥ 9,216 px².",
        "",
        "## Bounding-box area quantiles (px²)",
        "",
        "| Percentile | Area |", "|---:|---:|",
    ]
    lines.extend(f"| {percentile:.0%} | {_quantile(areas, percentile):.2f} |"
                 for percentile in (0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1))
    lines.extend(["", "## COCO size groups", "", "| Group | Area (px²) | Boxes | Box share | Images containing group |",
                  "|---|---:|---:|---:|---:|"])
    for row in size_rows:
        upper = "∞" if row["area_max_px2_exclusive"] == "inf" else f"<{row['area_max_px2_exclusive']:,.0f}"
        lower = f"≥{row['area_min_px2_inclusive']:,.0f}" if row["size"] == "large" else f"{row['area_min_px2_inclusive']:,.0f}"
        interval = f"{lower}–{upper}" if row["size"] != "large" else lower
        lines.append(f"| {row['size']} | {interval} | {row['bbox_count']:,} | {row['bbox_fraction']:.1%} | {row['image_count']:,} |")
    lines.extend(["", "## Class distribution", "",
                  "Image counts are not mutually exclusive: an image can contain multiple classes.",
                  "", "| Class | Bboxes | Share of bboxes | Images containing class |", "|---|---:|---:|---:|"])
    for row in class_rows:
        lines.append(f"| {row['class_name']} | {row['bbox_count']:,} | {row['bbox_fraction']:.1%} | {row['image_count']:,} |")
    lines.extend(["", "## Bbox size by class", "",
                  "Counts are bbox counts; percentages show the size-group share within each class.",
                  "", "| Class | Small | Medium | Large |", "|---|---:|---:|---:|"])
    for category_id, class_name in sorted(classes.items()):
        cells = []
        for size_name in ("small", "medium", "large"):
            row = class_size_lookup[(category_id - 1, size_name)]
            cells.append(f"{row['bbox_count']:,} ({row['class_bbox_fraction']:.1%})")
        lines.append(f"| {class_name} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Images by street", "",
                  "| Street ID | Images | Share of images | Bboxes |", "|---:|---:|---:|---:|"])
    for row in street_rows:
        lines.append(f"| {row['street_id']} | {row['image_count']:,} | {row['image_fraction']:.1%} | {row['bbox_count']:,} |")
    lines.extend(["", "## Bboxes per image", "",
                  f"Images have {min(image_bbox_counts)}–{max(image_bbox_counts)} bboxes; mean {sum(image_bbox_counts) / len(image_bbox_counts):.2f}, median {_quantile(image_bbox_counts, 0.5):.0f}.",
                  "", "| Bboxes in image | Number of images |", "|---:|---:|"])
    lines.extend(f"| {count} | {image_bbox_histogram[count]:,} |" for count in sorted(image_bbox_histogram))
    lines.extend(["", "## Ground-truth bbox overlap", "",
                  "IoU is computed between distinct bbox pairs in the same image. Pair fractions use all same-image pairs as the denominator; image fractions use all labeled images. These thresholds describe rectangle overlap and do not establish visual occlusion.",
                  "", "| IoU threshold | Overlapping pairs | Pair fraction | Images with overlap | Image fraction |",
                  "|---:|---:|---:|---:|---:|"])
    for row in overlap_threshold_rows:
        lines.append(f"| {row['comparison']} {row['threshold']:.2f} | {row['overlapping_pairs']:,} / {row['total_pairs_same_image']:,} | {row['pair_fraction']:.2%} | {row['images_with_overlap']:,} / {len(images):,} | {row['image_fraction']:.2%} |")
    if overlap_pairs:
        lines.extend(["", "Positive-IoU pair summary: "
                      f"{len(overlap_pairs):,} pairs; IoU median {_quantile(overlap_ious, 0.5):.3f}, "
                      f"95th percentile {_quantile(overlap_ious, 0.95):.3f}, "
                      f"exact-IoU-1 pairs {sum(iou == 1 for iou in overlap_ious):,}.",
                      "", "Most frequent overlapping class pairs (IoU > 0):",
                      "Counts include every pair with any positive-area intersection; they are not filtered at the 0.1, 0.5, or 0.75 thresholds above.",
                      "", "| Class pair | Pairs with IoU > 0 |", "|---|---:|"])
        top_class_pairs = sorted(overlap_class_pair_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
        for (first_id, second_id), count in top_class_pairs:
            label = classes[first_id + 1] if first_id == second_id else f"{classes[first_id + 1]} × {classes[second_id + 1]}"
            lines.append(f"| {label} | {count:,} |")
        lines.extend(["", "Highest-IoU bbox pairs for visual review:", "",
                      "| Image ID | Street ID | Class pair | IoU |", "|---:|---:|---|---:|"])
        for pair in overlap_pairs[:10]:
            label = pair["first_class_name"] if pair["first_class_name"] == pair["second_class_name"] else f"{pair['first_class_name']} × {pair['second_class_name']}"
            lines.append(f"| {pair['image_id']} | {pair['street_id']} | {label} | {pair['iou']:.3f} |")
    lines.extend(["", "## Outputs", "", "- `class_counts.csv`: bbox count and image count per class.",
                  "- `street_counts.csv`: images and boxes per street.",
                  "- `box_size_distribution.csv`: COCO size bins.",
                  "- `class_size_counts.csv`: size-bin counts within each class.",
                  "- `image_stats.csv`: per-image counts and bbox-area summary.",
                  "- `overlap_thresholds.csv`: pair and image overlap rates at each IoU threshold.",
                  "- `overlap_pairs.csv`: every positive-IoU bbox pair after dedup, sorted by IoU descending.",
                  "- `duplicate_annotations.csv`: exact same-image, same-class bbox duplicates and source rows.",
                  "- `cross_class_bbox_conflicts.csv`: identical bbox coordinates assigned different classes; retained in EDA and overlap pending review.", ""])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return {"images": len(images), "raw_annotations": len(raw_annotations), "duplicates_removed": duplicate_annotations_removed,
            "boxes": total_boxes, "classes": len(classes), "cross_class_conflicts": len(cross_class_rows),
            "streets": len(street_images), "output_dir": output_dir}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="archive/za_traffic_2020",
                        help="Dataset root containing traffic_train/annotation.csv")
    parser.add_argument("--output-dir", default="outputs/eda", help="Directory for EDA outputs")
    args = parser.parse_args()
    result = analyze(args.data_root, args.output_dir)
    print(f"EDA complete: {result['images']} images, {result['raw_annotations']} raw annotations, "
          f"removed {result['duplicates_removed']} duplicates; {result['boxes']} unique annotations, "
          f"{result['classes']} classes, {result['streets']} streets -> {result['output_dir']}")


if __name__ == "__main__":
    main()
