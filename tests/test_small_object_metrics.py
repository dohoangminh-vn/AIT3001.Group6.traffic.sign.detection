import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from src.small_object_metrics import _load_validation_ground_truth, _precision_metrics, _run_coco_area_eval


class SmallObjectMetricTests(unittest.TestCase):
    def test_precision_tensor_ap50_and_ap50_95_ignore_absent_classes(self):
        # Dimensions: IoU threshold x recall x class x area x maxDet.
        precision = [
            [[[[0.5]]], [[[0.7]]]],
            [[[[0.9]]], [[[-1.0]]]],
        ]
        ap50, ap50_95 = _precision_metrics(precision, [0.5, 0.75])
        self.assertAlmostEqual(ap50, 0.6)
        self.assertAlmostEqual(ap50_95, 0.7)

    def test_validation_ground_truth_deduplicates_same_class_and_keeps_cross_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data"
            annotation = dataset / "traffic_train" / "annotation.csv"
            annotation.parent.mkdir(parents=True)
            fields = ["image_id", "street_id", "category_id", "supercategory", "width", "height", "bbox", "file_name"]
            with annotation.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                rows = [
                    (1, 1, "A", "[0, 0, 8, 8]"),
                    (1, 1, "A", "[0, 0, 8, 8]"),  # duplicate row
                    (1, 2, "B", "[0, 0, 8, 8]"),  # same geometry, distinct label
                    (2, 1, "A", "[0, 0, 16, 16]"),
                    (3, 1, "A", "[0, 0, 8, 16]"),  # area exactly 128
                ]
                for image_id, category_id, name, bbox in rows:
                    writer.writerow({"image_id": image_id, "street_id": 1, "category_id": category_id,
                                     "supercategory": name, "width": 100, "height": 100, "bbox": bbox,
                                     "file_name": f"{image_id}.png"})
            prepared = root / "prepared"
            prepared.mkdir()
            (prepared / "split.json").write_text(json.dumps({"splits": {"val": [1, 2, 3]}}), encoding="utf-8")

            ground_truth, images = _load_validation_ground_truth(dataset, prepared)

            self.assertEqual(len(images), 3)
            self.assertEqual(len(ground_truth["annotations"]), 4)
            self.assertEqual([row["category_id"] for row in ground_truth["annotations"]], [0, 1, 0, 0])
            self.assertEqual(sum(row["area"] < 128 for row in ground_truth["annotations"]), 2)
            self.assertEqual(sum(row["area"] < 256 for row in ground_truth["annotations"]), 3)

    def test_area_range_uses_strict_upper_cutoff(self):
        observed_ranges = []

        class FakeCOCO:
            def __init__(self):
                self.dataset = {}

            def createIndex(self):
                pass

        class FakeCOCOeval:
            def __init__(self, ground_truth, detections, iou_type):
                self.params = SimpleNamespace(imgIds=[], areaRng=[], areaRngLbl=[], iouThrs=[0.5, 0.75])
                self.eval = {"precision": [[[[[0.5]]], [[[0.7]]]], [[[[0.9]]], [[[-1.0]]]] ]}

            def evaluate(self):
                observed_ranges.extend(self.params.areaRng)

            def accumulate(self):
                pass

        coco_package = ModuleType("pycocotools")
        coco_module = ModuleType("pycocotools.coco")
        coco_module.COCO = FakeCOCO
        eval_module = ModuleType("pycocotools.cocoeval")
        eval_module.COCOeval = FakeCOCOeval
        with patch.dict(sys.modules, {"pycocotools": coco_package,
                                      "pycocotools.coco": coco_module,
                                      "pycocotools.cocoeval": eval_module}):
            ap50, ap50_95 = _run_coco_area_eval(
                {"images": [{"id": 1}], "annotations": [], "categories": []}, [], 128
            )
        self.assertEqual(ap50, 0.6)
        self.assertAlmostEqual(ap50_95, 0.7)
        self.assertLess(observed_ranges[0][1], 128.0)


if __name__ == "__main__":
    unittest.main()
