"""CPU-only plots, offline HTML and an explicit report-file allowlist."""

import csv
import html
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from src.analysis import read_json, write_csv, write_json


def rows(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    return float(value) if value not in (None, "") else float("nan")


def _plot(path, title, labels, series, xlabel="", ylabel="", bars=False, selected=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for index, (name, values) in enumerate(series.items()):
        if bars:
            width = 0.8 / len(series)
            ax.bar([i - 0.4 + width * (index + 0.5) for i in range(len(labels))], values, width=width, label=name)
        else:
            ax.plot(labels, values, label=name, linewidth=2)
    if bars:
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_ylim(bottom=0)
    if ylabel in {"AP", "Score", "Validation score", "Precision", "Recall", "F1"} or ylabel.startswith(("AP;", "Recall;")):
        ax.set_ylim(0, 1.05)
    if selected is not None:
        ax.axvline(selected, color="#222222", linestyle="--", alpha=0.6, label=f"Selected: {selected:g}")
    ax.grid(axis="y", alpha=0.2)
    if series:
        ax.legend(fontsize="small")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _evaluation_plots(directory):
    operating = {row["class_id"]: row for row in rows(directory / "operating_metrics.csv")}
    ap = rows(directory / "class_ap.csv")
    if ap:
        labels = [f"{row['class_name']} (n={operating.get(row['class_id'], {}).get('gt_count', '?')})" for row in ap]
        _plot(directory / "class_ap.png", "AP by class — Ultralytics", labels,
              {"AP50": [number(row["ap50"]) for row in ap], "AP50:95": [number(row["ap50_95"]) for row in ap]}, ylabel="AP", bars=True)
    native = defaultdict(lambda: defaultdict(list))
    axes = {}
    for row in rows(directory / "native_curves.csv"):
        native[row["curve"]][row["class_name"]].append((number(row["x"]), number(row["y"])))
        axes[row["curve"]] = row["x_label"], row["y_label"]
    for curve, classes in native.items():
        filename = re.sub(r"[^a-zA-Z0-9_-]", "_", curve)
        points = next(iter(classes.values()))
        _plot(directory / f"{filename}.png", f"{curve} — Ultralytics (class curves)",
              [point[0] for point in points], {name: [point[1] for point in values] for name, values in classes.items()}, *axes[curve])
    confidence = rows(directory / "confidence_metrics.csv")
    if confidence:
        _plot(directory / "operating_confidence.png", "Operating metrics — micro average, fixed matching IoU",
              [number(row["confidence"]) for row in confidence],
              {key: [number(row[key]) for row in confidence] for key in ("precision", "recall", "f1")}, "Confidence", "Score",
              selected=read_json(directory / "metrics.json")["confidence"])
    small = rows(directory / "small_object_metrics.csv")
    if small:
        _plot(directory / "small_object_ap.png", "Small-object AP — cumulative, overlapping ranges (original pixels)",
              [f"<{row['area_cutoff_px2_exclusive']} px²\nn={row['gt_bbox_count']}" for row in small],
              {"COCO AP50": [number(row["ap50"]) for row in small], "COCO AP50:95": [number(row["ap50_95"]) for row in small]}, ylabel="AP; empty groups = N/A", bars=True)
    sizes = rows(directory / "size_recall.csv")
    if sizes:
        labels = [f"[{row['area_min_px2']}, {row['area_max_px2_exclusive'] or '∞'})\nn={row['gt_count']}" for row in sizes]
        _plot(directory / "size_recall.png", "Recall by disjoint area ranges (original pixels)", labels,
              {"Recall": [number(row["recall"]) for row in sizes]}, ylabel="Recall; empty groups = N/A", bars=True)
        _plot(directory / "size_counts.png", "Ground-truth count by disjoint area ranges", labels,
              {"Ground truths": [number(row["gt_count"]) for row in sizes]}, ylabel="Boxes", bars=True)
    matrix = rows(directory / "confusion_matrix.csv")
    if matrix:
        import matplotlib.pyplot as plt
        labels = list(matrix[0])[1:]
        for normalized in (False, True):
            values = [[number(row[label]) for label in labels] for row in matrix]
            if normalized:
                values = [[value / sum(row) if sum(row) else 0 for value in row] for row in values]
            fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
            ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
            maximum = max(max(row) for row in values)
            for i, row in enumerate(values):
                for j, value in enumerate(row):
                    ax.text(j, i, f"{value:.2f}" if normalized else str(int(value)), ha="center", va="center", fontsize=8,
                            color="white" if value > maximum / 2 else "black")
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels)), labels)
            ax.set(xlabel="Predicted", ylabel="Ground truth", title="Operating confusion matrix" + (" — row normalized" if normalized else " — counts"))
            fig.savefig(directory / ("confusion_normalized.png" if normalized else "confusion_counts.png"), dpi=140)
            plt.close(fig)


def _tuning_plots(directory):
    best = read_json(directory / "best_thresholds.json")
    for key in ("nms_iou", "conf"):
        data = rows(directory / f"{key}_results.csv")
        if not data:
            continue
        metrics = [name for name in ("map50_95", "map50", "precision", "recall", "f1") if name in data[0]]
        _plot(directory / f"{key}_sweep.png", f"{key}: selected {best['selected_thresholds'][key]}",
              [number(row["threshold"]) for row in data], {name: [number(row[name]) for row in data] for name in metrics}, key, "Validation score",
              selected=best['selected_thresholds'][key])


def _diagnostic_plots(directory):
    errors = rows(directory / "errors.csv")
    counts = Counter((row["ground_truth_class"] or row["predicted_class"], row["error_type"]) for row in errors)
    labels = sorted({key[0] for key in counts})
    if labels:
        _plot(directory / "errors_by_class.png", "Diagnostic events — a class mistake is one event", labels,
              {kind: [counts[label, kind] for label in labels] for kind in ("false_positive", "false_negative", "misclassification")}, ylabel="Events", bars=True)
    from PIL import Image, ImageDraw
    legend = Image.new("RGB", (800, 90), "white")
    draw = ImageDraw.Draw(legend)
    for index, (label, color) in enumerate((("TP", "green"), ("FP", "red"), ("FN", "orange"), ("Wrong class", "purple"))):
        x = index * 195 + 10
        draw.rectangle((x, 20, x + 20, 40), fill=color)
        draw.text((x + 26, 24), label, fill="black")
    legend.save(directory / "legend.png")


def report_files(root):
    """Only known report locations; never include arbitrary run-root images or checkpoints."""
    root = Path(root).resolve()
    manifest = read_json(root / "run_manifest.json")
    candidates = [root / name for name in ("index.html", "run_manifest.json", "resolved_config.yaml", "overrides.yaml", "environment.json")]
    candidates += [root / "split" / name for name in ("split.json", "train.txt", "val.txt")]
    for name in ("training", "logs"):
        candidates.extend((root / name).rglob("*"))
    for entry in manifest.get("stages", {}).values():
        if entry.get("path"):
            directory = (root / entry["path"]).resolve()
            directory.relative_to(root)
            candidates.extend(directory.rglob("*"))
    if all(manifest.get("stages", {}).get(key, {}).get("status") == "complete"
           for key in ("evaluation/baseline", "evaluation/tuned")):
        candidates.extend((root / "evaluation").glob("comparison*"))
    allowed = {".csv", ".json", ".yaml", ".yml", ".png", ".jpg", ".jpeg", ".html", ".log"}
    result = []
    for path in candidates:
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        path.resolve().relative_to(root)
        if any(part in {"wandb", "downloads", "weights", "__pycache__"} for part in relative.parts):
            continue
        if path.name.lower().startswith("readme") or path.name in {"results.png", "train_batch0.jpg"}:
            continue
        if relative.parts[0] == "training" and path.suffix not in {".csv", ".yaml", ".log"}:
            continue
        if relative.parts[0] == "logs" and path.suffix != ".log":
            continue
        if path.suffix.lower() in allowed or relative.as_posix() in {"split/train.txt", "split/val.txt"}:
            result.append(path)
    return sorted(set(result))


def render_report(root):
    root = Path(root).resolve()
    manifest = read_json(root / "run_manifest.json")
    esc = html.escape

    def url(path):
        return quote(path.relative_to(root).as_posix())

    def figure(path):
        return f'<figure><a href="{url(path)}"><img loading="lazy" src="{url(path)}" alt="{esc(path.stem)}"></a><figcaption>{esc(path.stem)}</figcaption></figure>'

    sections = []
    evaluations = {}
    titles = {"evaluation/baseline": "Best model — baseline", "evaluation/tuned": "Best model — tuned",
              "tuning": "Threshold tuning", "diagnostics/baseline": "Detection errors — baseline", "diagnostics/tuned": "Detection errors — tuned"}
    for key, title in titles.items():
        entry = manifest.get("stages", {}).get(key, {})
        status = entry.get("status", "not_run")
        content = f'<p>Status: {esc(status)} {esc(entry.get("error", ""))}</p>'
        if status == "complete":
            directory = (root / entry["path"]).resolve()
            directory.relative_to(root)
            if key.startswith("evaluation"):
                _evaluation_plots(directory)
                summary = read_json(directory / "metrics.json")
                evaluations[key] = summary
                cards = {"mAP50": summary["ultralytics"]["map50"], "mAP50:95": summary["ultralytics"]["map50_95"],
                         **{name: summary["operating"][name] for name in ("precision", "recall", "f1")}}
                content += '<div class="cards">' + ''.join(
                    f'<div class="card"><span>{esc(label)}</span><strong>{value:.3f}</strong></div>' for label, value in cards.items()) + '</div>'
                content += f'<p>Validation images: {summary.get("validation_images", "N/A")}; operating confidence: {summary["confidence"]}; matching IoU: {summary.get("match_iou", "N/A")}</p>'
                content += '<details><summary>Evaluation settings and provenance</summary><pre>' + esc(json.dumps(summary, indent=2, ensure_ascii=False)) + '</pre></details>'
                content += '<p>Small-object AP uses cumulative overlapping ranges; recall uses disjoint ranges. Empty groups: N/A. AP confidence = 0.001.</p>'
            elif key == "tuning":
                _tuning_plots(directory)
                content += '<p>Thresholds were selected on validation; these are not independent test results.</p>'
            else:
                _diagnostic_plots(directory)
                report = read_json(directory / "report.json")
                content += f'<p>Confidence: {report["confidence_threshold"]}; matching IoU: {report["matching_iou_threshold"]}; NMS IoU: {report["nms_iou_threshold"]}</p>'
                summaries = sorted(rows(directory / "image_summary.csv"), key=lambda row: -int(row["error_count"]))
                content += '<table><tr><th>Image</th><th>Error events</th><th>FP</th><th>FN</th><th>Wrong class</th></tr>'
                for row in summaries[:20]:
                    target = directory / row["overlay"] if row["overlay"] else None
                    label = f'<a href="{url(target)}">{esc(row["image"])}</a>' if target else esc(row["image"])
                    content += '<tr><td>' + label + '</td>' + ''.join(f'<td>{esc(row[field])}</td>' for field in ("error_count", "false_positives", "false_negatives", "misclassifications")) + '</tr>'
                content += '</table><div class="gallery">'
                for row in summaries:
                    if row["overlay"]:
                        content += f'<figure class="detection" data-classes="{esc(row.get("classes", ""), quote=True)}" data-types="{esc(row.get("error_types", "") or "correct", quote=True)}">'
                        content += f'<a href="{url(directory / row["overlay"])}"><img loading="lazy" src="{url(directory / row["overlay"])}"></a><figcaption>{esc(row["image"])} — {esc(row["error_count"])} errors</figcaption></figure>'
                content += '</div>'
            charts = sorted(directory.glob("*.png"))
            if key.startswith("evaluation"):
                small = {"small_object_ap.png", "size_recall.png", "size_counts.png"}
                content += '<h3>Detection quality by class</h3>' + ''.join(figure(path) for path in charts if path.name not in small)
                content += '<h3>Small traffic signs</h3>' + ''.join(figure(path) for path in charts if path.name in small)
            else:
                content = ''.join(figure(path) for path in charts) + content
        sections.append(f'<section><h2>{esc(title)}</h2>{content}</section>')
    if len(evaluations) == 2:
        baseline, tuned = evaluations["evaluation/baseline"], evaluations["evaluation/tuned"]
        comparison = [{"variant": label, **data["ultralytics"], **data["operating"]} for label, data in (("baseline", baseline), ("tuned", tuned))]
        write_csv(root / "evaluation" / "comparison.csv", comparison)
        for metrics, filename in ((("precision", "recall", "f1"), "comparison_scores.png"), (("fp", "fn"), "comparison_errors.png")):
            path = root / "evaluation" / filename
            _plot(path, "Baseline vs tuned — same validation split", list(metrics),
                  {row["variant"]: [number(row[key]) for key in metrics] for row in comparison}, bars=True)
            sections.append(figure(path))
    # Existing report files are linked, including CSVs, JSON and configs.
    files = report_files(root)
    links = ''.join(f'<li><a href="{url(path)}">{esc(path.relative_to(root).as_posix())}</a></li>' for path in files if path.name != "index.html")
    name = esc(manifest["run_name"])
    wandb_url = esc(manifest.get("wandb_url") or "", quote=True)
    upload_note = '' if manifest.get('model_upload') == 'complete' else '<p><strong>Model upload is not confirmed.</strong> Sync/upload to W&B before discarding local weights and context.</p>'
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name}</title><style>
body{{font:16px system-ui,sans-serif;background:#f3f5f8;color:#182335;max-width:1200px;margin:auto;padding:24px}}
section{{background:white;padding:20px;margin:20px 0;border-radius:12px}}h1{{overflow-wrap:anywhere}}a{{color:#175ac6}}
figure{{display:inline-block;vertical-align:top;width:46%;margin:1%;background:white}}img{{width:100%;height:auto}}figcaption{{padding:8px;overflow-wrap:anywhere}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:12px}}input,select{{padding:10px;margin:4px}}.hidden{{display:none}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}.card{{background:#eef3fb;border-radius:8px;padding:14px;min-width:110px}}.card strong{{display:block;font-size:28px}}.card span{{color:#52637a}}
@media(max-width:700px){{figure{{width:98%}}body{{padding:10px}}}}
</style><h1>{name}</h1><p><a href="{wandb_url}">Source W&B run</a></p><p>Weights: {esc(str(manifest.get('artifact_uri')))}</p>
<p>Training charts and weights are excluded from this report.</p>{upload_note}
<label>Gallery class: <input id="class-filter" placeholder="Class name"></label>
<label>Error: <select id="type-filter"><option value="">All</option><option>correct</option><option>false_positive</option><option>false_negative</option><option>misclassification</option></select></label>
{''.join(sections)}<section><h2>Files</h2><ul>{links}</ul></section><script>
function filterGallery(){{const c=document.getElementById('class-filter').value.toLowerCase();const t=document.getElementById('type-filter').value;
document.querySelectorAll('.detection').forEach(e=>e.classList.toggle('hidden',!e.dataset.classes.toLowerCase().includes(c)||(t&&!e.dataset.types.split('|').includes(t))));}}
document.getElementById('class-filter').addEventListener('input',filterGallery);document.getElementById('type-filter').addEventListener('change',filterGallery);
</script></html>'''
    (root / "index.html").write_text(document, encoding="utf-8")


def export_reports(run_dir, export_dir):
    root = Path(run_dir).expanduser().resolve()
    manifest = read_json(root / "run_manifest.json")
    training = root / "training"
    training.mkdir(exist_ok=True)
    for name in ("results.csv", "args.yaml"):
        if (root / name).is_file():
            import shutil
            shutil.copyfile(root / name, training / name)
    render_report(root)
    files = report_files(root)
    destination = Path(export_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^\w.-]+', '-', manifest["run_name"]).strip('.-') or "run"
    archive = destination / f"{safe_name}-reports-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:6]}.zip"
    manifest.setdefault("exports", []).append({"filename": archive.name, "created_at": datetime.now(timezone.utc).isoformat()})
    write_json(root / "run_manifest.json", manifest)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(set(files)):
            bundle.write(path, f"{safe_name}/{path.relative_to(root).as_posix()}")
    print(f"Run: {manifest['run_name']}\nZIP: {archive}\nSize: {archive.stat().st_size / 1024 ** 2:.2f} MiB")
    return archive
