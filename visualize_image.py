"""Draw ground-truth annotations for one ZA Traffic 2020 image ID.

Example:
    python visualize_image.py 9738
    python visualize_image.py 9738 --output outputs/eda/review_9738.png
"""

import argparse
import ast
import csv
from pathlib import Path

COLORS = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
)


def _font(size, ImageFont):
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _read_annotations(data_root):
    annotation_path = data_root / "traffic_train" / "annotation.csv"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {annotation_path}")
    grouped = {}
    with annotation_path.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), 2):
            try:
                bbox = tuple(map(float, ast.literal_eval(row["bbox"])))
                if len(bbox) != 4:
                    raise ValueError("bbox must have four coordinates")
                x, y, width, height = bbox
                if width <= 0 or height <= 0:
                    raise ValueError("bbox width and height must be positive")
                annotation = {
                    "file_name": row["file_name"],
                    "image_width": int(row["width"]),
                    "image_height": int(row["height"]),
                    "street_id": row["street_id"],
                    "category_id": int(row["category_id"]),
                    "class_name": row["supercategory"],
                    "box_id": row.get("box_id", ""),
                    "bbox": (x, y, width, height),
                    "box": (x, y, x + width, y + height),
                }
                grouped.setdefault(row["image_id"], []).append(annotation)
            except (KeyError, TypeError, ValueError, SyntaxError) as error:
                raise ValueError(f"{annotation_path}, row {line_number}: {error}") from error
    return grouped


def _render_image(image_id, selected, data_root, output_path):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError("Pillow is required; install project dependencies with `python -m pip install -r requirements.txt`") from error

    if not selected:
        raise ValueError(f"No labeled image found with image_id={image_id}")
    metadata = {(row["file_name"], row["image_width"], row["image_height"]) for row in selected}
    if len(metadata) != 1:
        raise ValueError(f"Inconsistent image metadata for image_id={image_id}")
    file_name, expected_width, expected_height = next(iter(metadata))
    image_path = data_root / "traffic_train" / "images" / file_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    if image.size != (expected_width, expected_height):
        raise ValueError(f"Image dimensions {image.size} disagree with annotations "
                         f"({expected_width}, {expected_height}) for image_id={image_id}")

    draw = ImageDraw.Draw(image)
    line_width = max(2, min(image.size) // 250)
    font = _font(max(12, min(image.size) // 48), ImageFont)
    for index, row in enumerate(selected, 1):
        color = COLORS[(row["category_id"] - 1) % len(COLORS)]
        box = row["box"]
        draw.rectangle(box, outline=color, width=line_width)
        # Put a small index at the box corner; the full labels go into a
        # separate legend below the image so coincident boxes stay readable.
        marker = str(index)
        left, top, right, bottom = draw.textbbox((0, 0), marker, font=font)
        marker_width, marker_height = right - left, bottom - top
        x1, y1 = box[0], max(0, box[1] - marker_height - 5)
        draw.rectangle((x1, y1, x1 + marker_width + 6, y1 + marker_height + 4), fill=color)
        draw.text((x1 + 3, y1 + 1), marker, fill="white", font=font)

    legend_line_height = max(20, font.size + 8)
    legend_height = 34 + len(selected) * legend_line_height
    canvas = Image.new("RGB", (image.width, image.height + legend_height), "white")
    canvas.paste(image, (0, 0))
    legend = ImageDraw.Draw(canvas)
    legend_top = image.height + 8
    legend.text((10, legend_top), f"Image {image_id}: {len(selected)} annotations", fill="black", font=font)
    for index, row in enumerate(selected, 1):
        color = COLORS[(row["category_id"] - 1) % len(COLORS)]
        x, y, width, height = row["bbox"]
        box_id = f"box_id={row['box_id']}" if row["box_id"] else "box_id=unknown"
        label = (f"{index}. {row['class_name']} ({box_id})  "
                 f"bbox=[{x:g}, {y:g}, {width:g}, {height:g}]")
        line_y = legend_top + 22 + (index - 1) * legend_line_height
        legend.rectangle((10, line_y + 4, 22, line_y + 16), fill=color)
        legend.text((30, line_y), label, fill="black", font=font)

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Rendered image_id={image_id}: {len(selected)} annotation rows -> {output_path}")
    return output_path


def visualize(image_id, data_root="archive/za_traffic_2020", output=None):
    """Render all raw annotation rows for image_id and save a PNG."""
    data_root = Path(data_root).expanduser().resolve()
    grouped = _read_annotations(data_root)
    image_id = str(image_id)
    if image_id not in grouped:
        raise ValueError(f"No labeled image found with image_id={image_id}")
    output_path = (Path(output).expanduser().resolve() if output else
                   Path("outputs/visualizations") / f"image_{image_id}.png")
    return _render_image(image_id, grouped[image_id], data_root, output_path)


def visualize_street(street_id, data_root="archive/za_traffic_2020", output_dir=None, limit=30):
    """Render images belonging to a street as individual PNG files."""
    data_root = Path(data_root).expanduser().resolve()
    grouped = _read_annotations(data_root)
    image_ids = sorted(
        (image_id for image_id, annotations in grouped.items()
         if any(row["street_id"] == str(street_id) for row in annotations)),
        key=lambda value: (int(value), value),
    )
    if not image_ids:
        raise ValueError(f"No labeled images found with street_id={street_id}")
    if limit < 0:
        raise ValueError("limit must be zero (all images) or a positive integer")
    selected_ids = image_ids[:limit] if limit else image_ids
    image_output_dir = (Path(output_dir).expanduser().resolve() if output_dir else
                        Path("outputs/visualizations") / f"street_{street_id}")
    image_output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for image_id in selected_ids:
        filename = f"image_{image_id}.png"
        outputs.append(_render_image(image_id, grouped[image_id], data_root, image_output_dir / filename))
    print(f"Rendered {len(selected_ids)} of {len(image_ids)} street images -> {image_output_dir}")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_id", nargs="?", help="image_id from traffic_train/annotation.csv")
    parser.add_argument("--street-id", help="Render images for this street_id instead of one image")
    parser.add_argument("--data-root", default="archive/za_traffic_2020",
                        help="Dataset root containing traffic_train/")
    parser.add_argument("--output", help="Output PNG path (default: outputs/visualizations/image_<id>.png)")
    parser.add_argument("--output-dir", help="Directory for street PNGs (default: outputs/visualizations/street_<id>)")
    parser.add_argument("--limit", type=int, default=30,
                        help="Maximum street images to render; use 0 for all (default: 30)")
    args = parser.parse_args()
    if bool(args.image_id) == bool(args.street_id):
        parser.error("provide either image_id or --street-id")
    if args.street_id:
        if args.output:
            parser.error("--output is for one image; use --output-dir with --street-id")
        visualize_street(args.street_id, args.data_root, args.output_dir, args.limit)
    else:
        visualize(args.image_id, args.data_root, args.output)


if __name__ == "__main__":
    main()
