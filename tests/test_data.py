import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data import prepare, read_annotations, split_by_street, yolo_box


class DataTests(unittest.TestCase):
    def test_known_bbox_and_invalid_boxes(self):
        self.assertEqual(yolo_box([10, 20, 20, 10], 100, 50), (0.2, 0.5, 0.2, 0.2))
        for box in ([-1, 0, 1, 1], [0, 0, 0, 1], [99, 0, 2, 1], [float("nan"), 0, 1, 1]):
            with self.assertRaises(ValueError):
                yolo_box(box, 100, 50)

    def test_split_by_street_is_deterministic_and_keeps_groups_intact(self):
        images = {}
        image_id = 0
        for street_id, image_count in enumerate((2, 3, 5, 7, 11, 13)):
            for index in range(image_count):
                images[image_id] = {"street_id": street_id,
                                    "labels": [(index % 2, 0.5, 0.5, 0.1, 0.1)]}
                image_id += 1
        first = split_by_street(images, seed=42, val_fraction=0.2)
        second = split_by_street(images, seed=42, val_fraction=0.2)
        train, val, train_streets, val_streets = first
        self.assertEqual(first, second)
        self.assertFalse(set(train_streets) & set(val_streets))
        self.assertEqual(set(train) | set(val), set(images))
        self.assertFalse(set(train) & set(val))
        self.assertLess(abs(len(val) / len(images) - 0.2), 0.03)
        for image_id in val:
            self.assertIn(images[image_id]["street_id"], val_streets)
        for image_id in train:
            self.assertIn(images[image_id]["street_id"], train_streets)

    def test_prepare_mapping_reuse_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            image_dir = root / "traffic_train" / "images"
            image_dir.mkdir(parents=True)
            csv_path = image_dir.parent / "annotation.csv"
            columns = ["image_id", "street_id", "category_id", "box_id", "file_name", "width", "height", "bbox", "supercategory"]
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for i in range(1, 8):
                    Image.new("RGB", (100, 50)).save(image_dir / f"{i}.png")
                    writer.writerow(dict(image_id=i, street_id=i, category_id=i, box_id=i, file_name=f"{i}.png",
                                         width=100, height=50, bbox="[10, 20, 20, 10]", supercategory=f"class{i}"))
                # Exact same-class duplicate should collapse; same geometry with
                # a different class should remain as a flagged label conflict.
                writer.writerow(dict(image_id=1, street_id=1, category_id=1, box_id=8, file_name="1.png",
                                     width=100, height=50, bbox="[10, 20, 20, 10]", supercategory="class1"))
                writer.writerow(dict(image_id=1, street_id=1, category_id=2, box_id=9, file_name="1.png",
                                     width=100, height=50, bbox="[10, 20, 20, 10]", supercategory="class2"))
            before = {p.name: p.read_bytes() for p in image_dir.iterdir()}
            original_csv = csv_path.read_bytes()
            output = Path(directory) / "prepared"
            data_yaml = prepare(root, output)
            images, names, stats = read_annotations(root, return_stats=True)
            self.assertEqual(len(names), 7)
            self.assertEqual(stats["raw_annotation_count"], 9)
            self.assertEqual(stats["unique_annotation_count"], 8)
            self.assertEqual(stats["duplicate_annotations_removed"], 1)
            self.assertEqual(stats["duplicate_groups"], 1)
            self.assertEqual(stats["cross_class_bbox_conflicts"], 1)
            self.assertEqual(len(images[1]["labels"]), 2)
            manifest = json.loads((output / "split.json").read_text())
            self.assertEqual(manifest["preparation"]["split_policy"], "street_grouped_local_search_v1")
            self.assertEqual(set(manifest["street_ids"]["train"]) | set(manifest["street_ids"]["val"]), set(range(1, 8)))
            self.assertFalse(set(manifest["street_ids"]["train"]) & set(manifest["street_ids"]["val"]))
            self.assertEqual(manifest["split_stats"]["val_image_count"], len(manifest["splits"]["val"]))
            for subset, ids in manifest["splits"].items():
                for i in ids:
                    label = output / "labels" / subset / f"{i}.txt"
                    lines = label.read_text().splitlines()
                    self.assertEqual(len(lines), 2 if i == 1 else 1)
                    values = lines[0].split()
                    self.assertEqual(int(values[0]), i - 1)
                    self.assertEqual(tuple(map(float, values[1:])), (0.2, 0.5, 0.2, 0.2))
                    self.assertEqual(images[i]["labels"][0][0], i - 1)
            timestamp = (output / "split.json").stat().st_mtime_ns
            self.assertEqual(prepare(root, output), data_yaml)
            manifest = json.loads((output / "split.json").read_text())
            self.assertEqual(manifest["annotation_stats"]["duplicate_annotations_removed"], 1)
            self.assertEqual(timestamp, (output / "split.json").stat().st_mtime_ns)
            self.assertEqual(before, {p.name: p.read_bytes() for p in image_dir.iterdir()})
            self.assertEqual(original_csv, csv_path.read_bytes())
            with self.assertRaisesRegex(ValueError, "differs"):
                prepare(root, output, seed=1)
            with self.assertRaisesRegex(ValueError, "dedup policy"):
                prepare(root, output, seed=42, val_fraction=0.3)
            with self.assertRaisesRegex(ValueError, "outside"):
                prepare(root, root / "prepared")
            (image_dir / "1.png").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare(root, output)


if __name__ == "__main__":
    unittest.main()
