import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from src.analysis import (
    analysis_stage, prepare_analysis, publish_best, read_json, resolve_analysis,
    sha256, split_hash, write_json,
)


def config():
    cfg = OmegaConf.load("configs/config.yaml")
    del cfg.hydra
    del cfg.defaults
    cfg.model = OmegaConf.load("configs/model/yolov8.yaml")
    return cfg


def source_fixture(root):
    root = Path(root)
    (root / "split").mkdir(parents=True)
    saved = config()
    saved.data.root = "/original/kaggle/dataset"
    saved.model.imgsz = 960
    saved.seed = 17
    OmegaConf.save(saved, root / "resolved_config.yaml", resolve=True)
    OmegaConf.save(OmegaConf.create([]), root / "overrides.yaml")
    write_json(root / "environment.json", {"python": "3.11"})
    write_json(root / "run_manifest.json", {
        "run_name": "source-run", "run_id": "abc", "entity": "team", "project": "project",
        "wandb_url": "https://wandb.ai/team/project/runs/abc", "stages": {}, "backend": "yolo",
    })
    split = {"preparation": {"csv_sha256": "csv", "seed": 17}, "splits": {"val": [1, 2]}}
    write_json(root / "split" / "split.json", split)
    (root / "split" / "train.txt").write_text("3.png\n")
    (root / "split" / "val.txt").write_text("1.png\n2.png\n")
    (root / "best.pt").write_bytes(b"best weights")
    return split


class AnalysisTests(unittest.TestCase):
    def test_wandb_model_resolves_version_and_inherits_config_with_cli_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "artifact", Path(temporary) / "output"
            expected = source_fixture(source)
            cfg = config()
            cfg.analysis.checkpoint = "wandb://team/project/run_abc_model:best"
            cfg.analysis.run_dir = str(output)
            cfg.analysis.cache_dir = str(Path(temporary) / "cache")
            artifact = SimpleNamespace(entity="team", project="project", name="run_abc_model:v2",
                                       download=MagicMock(return_value=str(source)))
            wandb = MagicMock()
            wandb.Api.return_value.artifact.return_value = artifact
            with patch.dict("sys.modules", {"wandb": wandb}), patch("src.analysis.task_overrides", return_value=[
                "data.root=/new/dataset", "data.prepared_dir=/new/prepared", "evaluation.batch_size=2",
            ]):
                inherited, root, checkpoint, actual = resolve_analysis(cfg)
            self.assertEqual(inherited.model.imgsz, 960)
            self.assertEqual(inherited.seed, 17)
            self.assertEqual(inherited.data.root, "/new/dataset")
            self.assertEqual(inherited.evaluation.batch_size, 2)
            self.assertEqual(actual, expected)
            self.assertEqual(checkpoint, source / "best.pt")
            manifest = read_json(root / "run_manifest.json")
            self.assertEqual(manifest["artifact_uri"], "wandb://team/project/run_abc_model:v2")
            self.assertEqual(manifest["checkpoint_sha256"], sha256(checkpoint))
            wandb.Api.return_value.artifact.assert_called_once_with("team/project/run_abc_model:best", type="model")

    def test_threshold_file_rejects_other_checkpoint_or_inference_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = source_fixture(root)
            cfg = config()
            cfg.analysis.checkpoint = str(root / "best.pt")
            cfg.analysis.run_dir = str(root)
            thresholds = root / "thresholds.json"
            cfg.analysis.thresholds_file = str(thresholds)
            data = {"checkpoint_sha256": "wrong", "validation_split_sha256": split_hash(expected)}
            write_json(thresholds, data)
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                resolve_analysis(cfg)
            data.update(checkpoint_sha256=sha256(root / "best.pt"), imgsz=640, tile=False, max_detections=300)
            write_json(thresholds, data)
            with self.assertRaisesRegex(ValueError, "imgsz"):
                resolve_analysis(cfg)
            data.update(imgsz=960, selected_thresholds={"conf": 0.4, "nms_iou": 0.65}, metric_matching_iou=0.6)
            write_json(thresholds, data)
            actual, *_ = resolve_analysis(cfg)
            self.assertEqual(actual.diagnostic.conf, 0.4)
            self.assertEqual(actual.evaluation.inference.nms_iou, 0.65)
            self.assertEqual(actual.evaluation.match_iou, 0.6)
            with patch("src.analysis.task_overrides", return_value=["diagnostic.tile=640"]):
                with self.assertRaisesRegex(ValueError, "tile"):
                    resolve_analysis(cfg, "diagnostic")

    def test_prepared_split_must_match_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = config()
            expected = source_fixture(root)
            write_json(root / "split.json", {**expected, "splits": {"val": [99]}})
            with patch("src.data.prepare", return_value=root / "data.yaml"):
                with self.assertRaisesRegex(ValueError, "split differs"):
                    prepare_analysis(cfg, expected)

    def test_failed_stage_is_recorded_without_overwriting_prior_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_fixture(root)
            cfg = config()
            with analysis_stage(root, "evaluation", cfg) as first:
                pass
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                with analysis_stage(root, "evaluation", cfg) as second:
                    raise RuntimeError("CUDA out of memory")
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            entry = read_json(root / "run_manifest.json")["stages"]["evaluation/baseline"]
            self.assertEqual(entry["status"], "failed")
            self.assertIn("CUDA", entry["error"])

    def test_publish_waits_for_upload_before_advertising_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_fixture(root)
            wandb, run = MagicMock(), MagicMock()
            run.id, run.entity, run.project = "abc", "team", "project"
            run.log_artifact.return_value.name = "run_abc_model:v3"
            with patch.dict("sys.modules", {"wandb": wandb}):
                manifest = publish_best(root, root / "best.pt", run)
            run.log_artifact.return_value.wait.assert_called_once()
            self.assertEqual(manifest["artifact_uri"], "wandb://team/project/run_abc_model:v3")
            wandb.Artifact.return_value.add_dir.assert_called_once()

    def test_failed_upload_does_not_publish_success_uri(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_fixture(root)
            wandb, run = MagicMock(), MagicMock()
            run.id = "abc"
            run.log_artifact.return_value.wait.side_effect = RuntimeError("upload failed")
            with patch.dict("sys.modules", {"wandb": wandb}), self.assertRaisesRegex(RuntimeError, "upload failed"):
                publish_best(root, root / "best.pt", run)
            self.assertFalse(read_json(root / "run_manifest.json").get("artifact_uri"))


if __name__ == "__main__":
    unittest.main()
