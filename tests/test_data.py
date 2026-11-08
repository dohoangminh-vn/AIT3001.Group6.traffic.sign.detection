import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data import prepare, read_annotations, split_images, yolo_box


class DataTests(unittest.TestCase):
    def test_known_bbox_and_invalid_boxes(self):
        self.assertEqual(yolo_box([10, 20, 20, 10], 100, 50), (0.2, 0.5, 0.2, 0.2))
        for box in ([-1, 0, 1, 1], [0, 0, 0, 1], [99, 0, 2, 1], [float("nan"), 0, 1, 1]):
            with self.assertRaises(ValueError):
                yolo_box(box, 100, 50)

    def test_split_by_unique_image(self):
        train, val = split_images(list(range(100)) * 3)
        self.assertEqual((len(train), len(val)), (80, 20))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train + val), set(range(100)))
        self.assertEqual((train, val), split_images(reversed(range(100))))

    def test_prepare_mapping_reuse_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            image_dir = root / "traffic_train" / "images"
            image_dir.mkdir(parents=True)
            csv_path = image_dir.parent / "annotation.csv"
            columns = ["image_id", "category_id", "box_id", "file_name", "width", "height", "bbox", "supercategory"]
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for i in range(1, 8):
                    Image.new("RGB", (100, 50)).save(image_dir / f"{i}.png")
                    writer.writerow(dict(image_id=i, category_id=i, box_id=i, file_name=f"{i}.png",
                                         width=100, height=50, bbox="[10, 20, 20, 10]", supercategory=f"class{i}"))
            before = {p.name: p.read_bytes() for p in image_dir.iterdir()}
            original_csv = csv_path.read_bytes()
            output = Path(directory) / "prepared"
            data_yaml = prepare(root, output)
            images, names = read_annotations(root)
            self.assertEqual(len(names), 7)
            manifest = json.loads((output / "split.json").read_text())
            for subset, ids in manifest["splits"].items():
                for i in ids:
                    label = output / "labels" / subset / f"{i}.txt"
                    values = label.read_text().split()
                    self.assertEqual(int(values[0]), i - 1)
                    self.assertEqual(tuple(map(float, values[1:])), (0.2, 0.5, 0.2, 0.2))
                    self.assertEqual(images[i]["labels"][0][0], i - 1)
            timestamp = (output / "split.json").stat().st_mtime_ns
            self.assertEqual(prepare(root, output), data_yaml)
            self.assertEqual(timestamp, (output / "split.json").stat().st_mtime_ns)
            self.assertEqual(before, {p.name: p.read_bytes() for p in image_dir.iterdir()})
            self.assertEqual(original_csv, csv_path.read_bytes())
            with self.assertRaisesRegex(ValueError, "differs"):
                prepare(root, output, seed=1)
            with self.assertRaisesRegex(ValueError, "outside"):
                prepare(root, root / "prepared")
            (image_dir / "1.png").unlink()
            with self.assertRaises(FileNotFoundError):
                prepare(root, output)


if __name__ == "__main__":
    unittest.main()
