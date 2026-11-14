import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from src.diagnostics import (
    match_detections,
    resolve_diagnostic_checkpoint,
    run_diagnostics,
)


class DetectionMatchingTests(unittest.TestCase):
    def test_same_class_overlap_is_true_positive(self):
        ground_truths = [{"class_id": 0, "box": (0, 0, 10, 10)}]
        predictions = [{"class_id": 0, "box": (0, 0, 10, 10), "confidence": 0.9}]

        result = match_detections(ground_truths, predictions)

        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["unmatched_ground_truths"], [])
        self.assertEqual(result["unmatched_predictions"], [])

    def test_wrong_class_overlap_is_one_misclassification(self):
        ground_truths = [{"class_id": 0, "box": (0, 0, 10, 10)}]
        predictions = [{"class_id": 1, "box": (0, 0, 10, 10), "confidence": 0.8}]

        result = match_detections(ground_truths, predictions)

        self.assertEqual(result["misclassifications"], 1)
        self.assertEqual(result["unmatched_ground_truths"], [])
        self.assertEqual(result["unmatched_predictions"], [])
        self.assertEqual(result["matches"][0]["error_type"], "misclassification")

    def test_duplicate_prediction_is_false_positive_and_missing_box_is_false_negative(self):
        ground_truths = [{"class_id": 0, "box": (0, 0, 10, 10)}]
        predictions = [
            {"class_id": 0, "box": (0, 0, 10, 10), "confidence": 0.9},
            {"class_id": 0, "box": (0, 0, 10, 10), "confidence": 0.7},
        ]

        duplicate = match_detections(ground_truths, predictions)
        missed = match_detections(ground_truths, [])

        self.assertEqual(duplicate["true_positives"], 1)
        self.assertEqual(duplicate["unmatched_predictions"], [1])
        self.assertEqual(missed["unmatched_ground_truths"], [0])

    def test_iou_threshold_is_validated(self):
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            match_detections([], [], iou_threshold=1.1)


class CheckpointResolutionTests(unittest.TestCase):
    def test_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"weights")

            result = resolve_diagnostic_checkpoint(str(checkpoint), directory)

            self.assertEqual(result, checkpoint.resolve())

    def test_wandb_checkpoint_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "artifact"
            artifact_dir.mkdir()
            checkpoint = artifact_dir / "best.pt"
            checkpoint.write_bytes(b"weights")
            wandb = MagicMock()
            wandb.Api.return_value.artifact.return_value.download.return_value = str(artifact_dir)

            with patch.dict("sys.modules", {"wandb": wandb}):
                result = resolve_diagnostic_checkpoint(
                    "entity/project/run_abc_checkpoint:latest", directory
                )

            self.assertEqual(result, checkpoint.resolve())
            wandb.Api.return_value.artifact.assert_called_once_with(
                "entity/project/run_abc_checkpoint:latest", type="checkpoint"
            )


class DiagnosticReportTests(unittest.TestCase):
    def test_mocked_prediction_writes_overlay_csv_and_report(self):
        class TensorLike:
            def __init__(self, value):
                self.value = value

            def cpu(self):
                return self

            def tolist(self):
                return self.value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            image_dir = prepared / "images" / "val"
            label_dir = prepared / "labels" / "val"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (100, 100), "white").save(image_dir / "sample.png")
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            data_yaml = prepared / "data.yaml"
            data_yaml.write_text(
                f"path: {prepared.as_posix()}\nval: images/val\nnames: [sign]\n",
                encoding="utf-8",
            )
            (prepared / "split.json").write_text(json.dumps({
                "preparation": {"csv_sha256": "abc", "seed": 42},
                "splits": {"val": [1]},
            }), encoding="utf-8")
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"weights")

            boxes = SimpleNamespace(
                xyxy=TensorLike([[25, 25, 75, 75], [0, 0, 10, 10]]),
                cls=TensorLike([0, 1]),
                conf=TensorLike([0.9, 0.7]),
            )
            model = MagicMock()
            model.predict.return_value = iter([
                SimpleNamespace(boxes=boxes, orig_shape=(100, 100))
            ])
            ultralytics = ModuleType("ultralytics")
            ultralytics.YOLO = MagicMock(return_value=model)
            cfg = SimpleNamespace(
                diagnostic=SimpleNamespace(
                    checkpoint=str(checkpoint), conf=0.25, match_iou=0.5, device="cpu"
                ),
                data=SimpleNamespace(root="unused", prepared_dir=str(prepared), val_fraction=0.2),
                model=SimpleNamespace(imgsz=640),
                output_dir=str(root / "outputs"),
                seed=42,
            )

            with patch("src.diagnostics.prepare", return_value=data_yaml), patch.dict(
                "sys.modules", {"ultralytics": ultralytics}
            ):
                output = run_diagnostics(cfg)

            self.assertTrue((output / "images" / "sample.jpg").is_file())
            with (output / "image_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                summary = next(csv.DictReader(stream))
            self.assertEqual(summary["true_positives"], "1")
            self.assertEqual(summary["false_positives"], "1")
            with (output / "errors.csv").open(encoding="utf-8-sig", newline="") as stream:
                errors = list(csv.DictReader(stream))
            self.assertEqual([row["error_type"] for row in errors], ["false_positive"])
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["totals"]["images"], 1)
            self.assertEqual(report["totals"]["false_positives"], 1)
            model.predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
