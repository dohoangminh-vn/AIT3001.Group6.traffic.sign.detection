import ast
import json
import re
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from urllib.parse import unquote

from PIL import Image

from src.analysis import read_json, write_csv, write_json
from src.reporting import export_reports
from test_analysis import source_fixture


def report_fixture(root):
    source_fixture(root)
    output = root / "evaluation" / "baseline" / "attempt"
    output.mkdir(parents=True)
    write_json(output / "metrics.json", {"confidence": 0.25, "ultralytics": {"map50": 0.8, "map50_95": 0.6},
                                         "operating": {"precision": 1, "recall": 0.5, "f1": 2/3, "fp": 0, "fn": 1}})
    write_csv(output / "class_ap.csv", [{"class_id": 0, "class_name": "sign", "ap50": 0.8, "ap50_95": 0.6}])
    write_csv(output / "operating_metrics.csv", [{"class_id": 0, "gt_count": 2}])
    write_csv(output / "small_object_metrics.csv", [{"area_cutoff_px2_exclusive": 128, "gt_bbox_count": 0, "ap50": None, "ap50_95": None}])
    write_csv(output / "size_recall.csv", [{"area_min_px2": 0, "area_max_px2_exclusive": 128, "gt_count": 0, "recall": None},
                                          {"area_min_px2": 128, "area_max_px2_exclusive": None, "gt_count": 2, "recall": 0.5}])
    write_csv(output / "confidence_metrics.csv", [{"confidence": 0.25, "precision": 1, "recall": 0.5, "f1": 2/3}])
    write_csv(output / "confusion_matrix.csv", [{"ground_truth": "sign", "sign": 1, "background": 1},
                                               {"ground_truth": "background", "sign": 0, "background": 0}])
    diag = root / "diagnostics" / "baseline" / "attempt"
    (diag / "images").mkdir(parents=True)
    Image.new("RGB", (80, 60), "white").save(diag / "images" / "1.jpg")
    write_json(diag / "report.json", {"confidence_threshold": 0.25, "matching_iou_threshold": 0.5, "nms_iou_threshold": 0.7})
    write_csv(diag / "image_summary.csv", [{"image": "1.png", "error_count": 1, "false_positives": 0, "false_negatives": 1,
                                          "misclassifications": 0, "overlay": "images/1.jpg", "classes": "sign", "error_types": "false_negative"}])
    write_csv(diag / "errors.csv", [{"ground_truth_class": "sign", "predicted_class": "", "error_type": "false_negative"}])
    manifest = read_json(root / "run_manifest.json")
    manifest["stages"] = {"training": {"status": "complete"},
                          "evaluation/baseline": {"status": "complete", "path": output.relative_to(root).as_posix()},
                          "diagnostics/baseline": {"status": "complete", "path": diag.relative_to(root).as_posix()}}
    write_json(root / "run_manifest.json", manifest)


class ReportTests(unittest.TestCase):
    def test_export_has_offline_links_and_excludes_weights_caches_training_charts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source-run"
            report_fixture(root)
            (root / "results.csv").write_text("epoch,loss\n1,1\n")
            (root / "results.png").write_bytes(b"training plot")
            cached = root / "evaluation" / "baseline" / "attempt" / "wandb"
            cached.mkdir()
            (cached / "secret.json").write_text('{}')
            archive = export_reports(root, Path(temporary) / "exports")
            with zipfile.ZipFile(archive) as bundle:
                names = bundle.namelist()
                self.assertIn("source-run/index.html", names)
                self.assertIn("source-run/split/split.json", names)
                self.assertIn("source-run/training/results.csv", names)
                self.assertTrue(any(name.endswith("class_ap.png") for name in names))
                self.assertTrue(any(name.endswith("images/1.jpg") for name in names))
                self.assertFalse(any(name.endswith(".pt") or "/wandb/" in name or name.endswith("results.png") or "README" in name for name in names))
                document = bundle.read("source-run/index.html").decode()
                for target in re.findall(r'(?:src|href)="([^"]+)"', document):
                    if target and not target.startswith("https://"):
                        self.assertIn("source-run/" + unquote(target), names)
                self.assertNotIn("cdn.", document)
                self.assertIn("not_run", document)
                self.assertIn("data-classes=", document)

    def test_partial_report_and_reexport_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source-run"
            source_fixture(root)
            first = export_reports(root, Path(temporary) / "exports")
            second = export_reports(root, Path(temporary) / "exports")
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_file())

    def test_failed_tuned_attempt_excludes_stale_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'source-run'
            source_fixture(root)
            (root / 'evaluation').mkdir()
            (root / 'evaluation' / 'comparison.csv').write_text('old comparison')
            manifest = read_json(root / 'run_manifest.json')
            manifest['stages']['evaluation/tuned'] = {'status': 'failed', 'error': 'CUDA out of memory'}
            write_json(root / 'run_manifest.json', manifest)
            archive = export_reports(root, Path(temporary) / 'exports')
            with zipfile.ZipFile(archive) as bundle:
                self.assertFalse(any(name.endswith('comparison.csv') for name in bundle.namelist()))
                self.assertIn('CUDA out of memory', bundle.read('source-run/index.html').decode())

    def test_notebook_executable_and_optional_commands_parse(self):
        notebook = json.loads(Path("notebooks/kaggle_pipeline.ipynb").read_text(encoding="utf-8"))
        commands = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = ''.join(cell["source"])
            tree = ast.parse(code)
            commands.extend(node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str))
            lines = code.splitlines()
            start = next((i for i, line in enumerate(lines) if re.match(r'# (?:RUN_DIR = train_run|run_command|RUN_NAME =)', line)), None)
            if start is not None:
                uncommented = '\n'.join(line[2:] if line.startswith('# ') else line for line in lines[start:])
                ast.parse(uncommented)
        for name in ("evaluate.py", "diagnose.py", "export_reports.py"):
            self.assertIn(name, commands)
            self.assertTrue(Path(name).is_file())

    def test_notebook_training_does_not_copy_other_runs_logs(self):
        notebook = json.loads(Path("notebooks/kaggle_pipeline.ipynb").read_text(encoding="utf-8"))
        code = next(''.join(cell['source']) for cell in notebook['cells'] if 'def train_run(' in ''.join(cell['source']))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session, output = root / 'session', root / 'runs'
            session.mkdir()
            output.mkdir()
            (session / 'previous-run.log').write_text('another run')
            namespace = dict(SESSION_DIR=session, OUTPUT_ROOT=output, json=json, shutil=shutil, subprocess=subprocess)
            exec(compile(code, '<notebook runner>', 'exec'), namespace)

            def fake_command(arguments, stage):
                run = output / 'new-run'
                run.mkdir()
                write_json(run / 'run_manifest.json', {'run_name': 'new-run'})
                (session / 'train-current.log').write_text('current run')

            namespace['run_command'] = fake_command
            selected = namespace['train_run']([])
            self.assertEqual(selected.name, 'new-run')
            self.assertEqual([path.name for path in (selected / 'logs').iterdir()], ['train-current.log'])


if __name__ == "__main__":
    unittest.main()
