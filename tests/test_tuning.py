import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.analysis import read_json
from src.tuning import _validate_config, run_tuning
from test_analysis import config, source_fixture


class TuningTests(unittest.TestCase):
    def test_confidence_sweep_reuses_cpu_predictions_and_records_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_fixture(root)
            cfg = config()
            cfg.analysis.checkpoint = str(root / "best.pt")
            cfg.analysis.run_dir = str(root)
            cfg.tuning.thresholds = ["conf"]
            cfg.tuning.conf["values"] = [0.1, 0.2, 0.5]
            records = [{"image": "1.png", "ground_truths": [{"class_id": 0, "box": [0, 0, 8, 8]}],
                        "predictions": [{"class_id": 0, "box": [0, 0, 8, 8], "confidence": 0.8},
                                        {"class_id": 0, "box": [30, 30, 40, 40], "confidence": 0.1}]}]
            ultralytics = SimpleNamespace(YOLO=MagicMock(), settings=MagicMock())
            with patch.dict("sys.modules", {"ultralytics": ultralytics}), \
                 patch("src.tuning.prepare_analysis", return_value="data.yaml"), \
                 patch("src.tuning.collect_predictions", return_value=(records, {0: "sign"})) as collect:
                output = run_tuning(cfg)
            collect.assert_called_once()
            result = read_json(output / "best_thresholds.json")
            self.assertEqual(result["selected_thresholds"]["conf"], 0.5)
            self.assertTrue(result["checkpoint_sha256"])
            self.assertTrue(result["validation_split_sha256"])
            self.assertEqual(result["imgsz"], 960)

    def test_nan_threshold_is_rejected(self):
        cfg = config()
        cfg.tuning.conf["values"] = [float("nan")]
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            _validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
