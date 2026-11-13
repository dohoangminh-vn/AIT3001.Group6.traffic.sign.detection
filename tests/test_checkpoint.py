"""Exercise artifact packaging without network/GPU; real resume is checked on Kaggle."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.yolo import checkpoint_callback, resolve_resume


class CheckpointTests(unittest.TestCase):
    def test_periodic_upload_and_resume_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "last.pt").write_bytes(b"optimizer checkpoint")
            (root / "best.pt").write_bytes(b"best checkpoint")
            trainer = SimpleNamespace(epoch=3, stop=False, wdir=root,
                                      last=root / "last.pt", best=root / "best.pt")
            run, wandb = MagicMock(), MagicMock()
            run.id = "abc123"
            info = {"run_id": "abc123", "project": "traffic", "entity": "team"}
            with patch.dict("sys.modules", {"wandb": wandb}):
                callback = checkpoint_callback(run, info, 5, True)
                callback(trainer)
                run.log_artifact.assert_not_called()
                trainer.epoch = 4
                callback(trainer)
                artifact = wandb.Artifact.return_value
                self.assertEqual(artifact.add_file.call_count, 3)
                run.log_artifact.assert_called_once_with(artifact, aliases=["latest"])
                run.log_artifact.return_value.wait.assert_called_once()
                api = wandb.Api.return_value
                previous = MagicMock(version="v0")
                current = MagicMock(version="v1")
                api.artifact_versions.return_value = [previous, current]
                # Trigger another save with explicit API version values.
                run.log_artifact.return_value.version = "v1"
                trainer.epoch, trainer.stop = 9, False
                callback(trainer)
                previous.delete.assert_called_once()
                current.delete.assert_not_called()
                api.artifact_versions.assert_called_with(
                    "checkpoint", "team/traffic/run_abc123_checkpoint"
                )
                path, metadata = resolve_resume(str(root / "last.pt"), root)
                self.assertEqual(path, (root / "last.pt").resolve())
                self.assertEqual(metadata, info)
                # The final partial interval is also protected before optimizer stripping.
                trainer.epoch, trainer.stop = 6, True
                callback(trainer)
                self.assertEqual(run.log_artifact.call_count, 3)

    def test_offline_checkpoint_does_not_delete_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "last.pt").write_bytes(b"optimizer checkpoint")
            trainer = SimpleNamespace(epoch=0, stop=False, wdir=root,
                                      last=root / "last.pt", best=root / "best.pt")
            run, wandb = MagicMock(), MagicMock()
            run.id = "abc123"
            with patch.dict("sys.modules", {"wandb": wandb}):
                checkpoint_callback(run, {"run_id": "abc123"}, 1, False)(trainer)
            wandb.Api.assert_not_called()

    def test_upload_failure_does_not_delete_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "last.pt").write_bytes(b"optimizer checkpoint")
            trainer = SimpleNamespace(epoch=0, stop=False, wdir=root,
                                      last=root / "last.pt", best=root / "best.pt")
            run, wandb = MagicMock(), MagicMock()
            run.id = "abc123"
            run.log_artifact.return_value.wait.side_effect = RuntimeError("upload failed")
            with patch.dict("sys.modules", {"wandb": wandb}):
                with self.assertRaisesRegex(RuntimeError, "upload failed"):
                    checkpoint_callback(run, {"run_id": "abc123"}, 1, True)(trainer)
            wandb.Api.assert_not_called()

    def test_artifact_download_and_missing_local_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "last.pt").write_bytes(b"checkpoint")
            (root / "run.json").write_text(json.dumps({"run_id": "abc123"}))
            wandb = MagicMock()
            wandb.Api.return_value.artifact.return_value.download.return_value = str(root)
            with patch.dict("sys.modules", {"wandb": wandb}):
                path, info = resolve_resume("team/project/model:latest", root)
                self.assertEqual(path, (root / "last.pt").resolve())
                self.assertEqual(info["run_id"], "abc123")
                wandb.Api.return_value.artifact.assert_called_with("team/project/model:latest", type="checkpoint")
                with self.assertRaises(FileNotFoundError):
                    resolve_resume("missing.pt", root)


if __name__ == "__main__":
    unittest.main()
