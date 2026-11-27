import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.analysis import read_json
from src.evaluation import _small_object_wandb_metrics, operating_metrics, run_evaluation
from src.small_object_metrics import log_small_object_metrics
from src.tuning import _map
from test_analysis import config, source_fixture


class OperatingMetricTests(unittest.TestCase):
    def test_small_object_history_includes_ap_and_counts_and_skips_undefined_ap(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "small.csv"
            path.write_text(
                "area_cutoff_px2_exclusive,gt_bbox_count,gt_image_count,ap50,ap50_95\n"
                "128,3,2,0.0,0.25\n"
                "256,0,0,,\n",
                encoding="utf-8",
            )
            values = _small_object_wandb_metrics(path)
        self.assertEqual(values["small_eval/area_lt_128/AP50"], 0.0)
        self.assertEqual(values["small_eval/area_lt_128/AP50_95"], 0.25)
        self.assertEqual(values["small_eval/area_lt_128/gt_bbox_count"], 3)
        self.assertEqual(values["small_eval/area_lt_256/gt_image_count"], 0)
        self.assertNotIn("small_eval/area_lt_256/AP50", values)

    def test_wrong_class_empty_predictions_and_size_denominators(self):
        records = [{"image": "1.png", "ground_truths": [
            {"class_id": 0, "box": [0, 0, 8, 8]},
            {"class_id": 1, "box": [20, 0, 36, 8]},
            {"class_id": 0, "box": [50, 0, 90, 40]},
        ], "predictions": [
            {"class_id": 0, "box": [0, 0, 8, 8], "confidence": 0.9},
            {"class_id": 0, "box": [20, 0, 36, 8], "confidence": 0.8},
        ]}]
        totals, classes, confusion, sizes, matches = operating_metrics(records, {0: "a", 1: "b"}, 0.25, 0.5, [128, 256, 512, 1024])
        self.assertEqual((totals["tp"], totals["fp"], totals["fn"]), (1, 1, 2))
        self.assertEqual(sum(row["gt_count"] for row in sizes), 3)
        self.assertEqual(sizes[0]["recall"], 1)
        self.assertEqual(sizes[1]["recall"], 0)
        self.assertIsNone(sizes[2]["recall"])
        self.assertEqual(sizes[-1]["gt_count"], 1)
        self.assertEqual(confusion[1][0], 1)
        self.assertEqual(confusion[0][-1], 1)
        self.assertEqual(len(matches), 3)
        missed, *_ = operating_metrics([{**records[0], "predictions": []}], {0: "a", 1: "b"}, 0.25, 0.5, [128])
        self.assertEqual(missed["fn"], 3)

    def test_area_cutoff_with_no_ground_truth_is_na(self):
        with tempfile.TemporaryDirectory() as temporary, patch("src.small_object_metrics._run_coco_area_eval") as evaluate:
            rows = log_small_object_metrics({"annotations": []}, [], Path(temporary) / "small.csv", None, [128])
            self.assertIsNone(rows[0]["ap50"])
            self.assertEqual(rows[0]["gt_bbox_count"], 0)
            evaluate.assert_not_called()

    def test_nms_map_sweep_forwards_tile_and_low_confidence(self):
        cfg = config()
        cfg.tuning.tile = 640
        model = MagicMock()
        model.val.return_value.box.map = 0.4
        model.val.return_value.box.map50 = 0.6
        _map(model, "data.yaml", cfg, 0.7, Path("report"))
        args = model.val.call_args.kwargs
        self.assertEqual(args["tile"], 640)
        self.assertEqual(args["conf"], 0.001)
        self.assertEqual(args["batch"], 1)

    def test_evaluate_keeps_ap_separate_from_operating_confidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = source_fixture(root)
            cfg = config()
            cfg.analysis.checkpoint = str(root / "best.pt")
            cfg.analysis.run_dir = str(root)
            cfg.evaluation.conf = 0.75
            cfg.evaluation.device = "cpu"
            cfg.tracking.mode = "offline"
            native = SimpleNamespace(box=SimpleNamespace(ap_class_index=[0], ap50=[0.8], ap=[0.5], map50=0.8, map=0.5),
                                     curves=["Precision-Recall(B)"], curves_results=[([0, 1], [[1, 0.5]], "Recall", "Precision")])
            model = MagicMock()
            model.val.return_value = native
            records = [{"image": "1.png", "ground_truths": [{"class_id": 0, "box": [0, 0, 8, 8]}],
                        "predictions": [{"class_id": 0, "box": [0, 0, 8, 8], "confidence": 0.5}]}]
            dataset = {"images": [{"id": 1, "file_name": "1.png"}], "annotations": [{"image_id": 1, "area": 64}], "categories": []}
            ultralytics = SimpleNamespace(YOLO=MagicMock(return_value=model), settings=MagicMock())
            torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            wandb = MagicMock()
            with patch.dict("sys.modules", {"ultralytics": ultralytics, "torch": torch, "wandb": wandb}), \
                 patch("src.evaluation.prepare_analysis", return_value="data.yaml"), \
                 patch("src.evaluation._load_dataset", return_value=([], None, {0: "a"})), \
                 patch("src.evaluation.collect_predictions", return_value=(records, {0: "a"})), \
                 patch("src.small_object_metrics._load_validation_ground_truth", return_value=(dataset, {})), \
                 patch("src.small_object_metrics.evaluate_coco_predictions", return_value=(0.8, 0.5)), \
                 patch("src.small_object_metrics._run_coco_area_eval", return_value=(0.8, 0.5)):
                output = run_evaluation(cfg)
            self.assertEqual(model.val.call_args.kwargs["conf"], 0.001)
            self.assertEqual(model.val.call_args.kwargs["batch"], 1)
            summary = read_json(output / "metrics.json")
            self.assertEqual(summary["operating"]["fn"], 1)
            self.assertEqual(summary["ultralytics"]["map50"], 0.8)
            self.assertTrue((output / "small_object_metrics.csv").is_file())
            wandb_run = wandb.init.return_value.__enter__.return_value
            logged = wandb_run.log.call_args.args[0]
            self.assertIn("small_eval/area_lt_128/AP50", logged)
            self.assertIn("small_eval/area_lt_128/gt_bbox_count", logged)
            self.assertEqual(logged["small_eval/area_lt_128/AP50"], 0.8)


if __name__ == "__main__":
    unittest.main()
