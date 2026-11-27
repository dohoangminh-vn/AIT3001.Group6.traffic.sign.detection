# Traffic sign detection — Kaggle training and offline reports

Hydra configures YOLOv8s/m/l and the optional Faster R-CNN baseline on ZA Traffic 2020. Each backend owns its training loop. GPU work runs on Kaggle; download a named report ZIP to view locally without Python or the dataset.

Use Python 3.10–3.13. The complete workflow is in [notebooks/kaggle_pipeline.ipynb](notebooks/kaggle_pipeline.ipynb). Push your implementation before cloning on Kaggle, enable Internet and a GPU, and grant access to the WANDB_API_KEY Kaggle Secret. Never save keys in YAML or Git.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Dataset and preparation

The dataset root contains traffic_train/annotation.csv, traffic_train/images/ and the unlabeled traffic_public_test/images/. Sources are read-only.

Preparation deduplicates identical same-class boxes and splits by street so every street belongs to one split. It saves CSV hash, seed, split policy, image/street IDs and class counts in split.json, plus source-relative train.txt/val.txt. Images are symlinked when supported, otherwise copied. Letterboxing happens online; no resized dataset is written.

Use a new data.prepared_dir when CSV, seed, validation fraction or split policy changes. Prepared image symlinks and absolute data.yaml paths are not portable report artifacts.

```bash
python train.py prepare_only=true \
  data.root=/kaggle/input/<dataset>/za_traffic_2020 \
  data.prepared_dir=/kaggle/working/prepared_traffic_signs
```

## Training

```bash
python train.py \
  model=yolov8 \
  data.root=/kaggle/input/<dataset>/za_traffic_2020 \
  data.prepared_dir=/kaggle/working/prepared_traffic_signs \
  output_dir=/kaggle/working/traffic_sign_runs \
  tracking.name=baseline-yolov8s \
  tracking.mode=online \
  augmentation.enabled=false \
  model.batch_size=16 \
  model.epochs=50 \
  evaluation.inference.score_threshold=0.001
```

Recipes are yolov8 (s), yolov8m and yolov8l. Batch 16 is a starting point, not a VRAM guarantee; reduce it for larger models. Current repository config enables augmentation; the notebook baseline explicitly disables it. YOLO recipes use workers 4, 640-pixel input, 50 epochs and patience 10. Null train arguments inherit pinned Ultralytics defaults. With optimizer=auto, Ultralytics can select its own learning rate.

Each unique run saves resolved config, overrides, split snapshots, package/GPU/Git metadata and run_manifest.json. Training retains validation for best selection and early stopping. **Separate small-object evaluation no longer runs inside the YOLO training process.**

At completion, the workflow publishes best.pt and run context together in the W&B model collection run_<id>_model, waits for upload, and prints its immutable versioned URI. Use that printed URI: native Ultralytics may also have published a weights-only version in the collection. Model blobs are managed by W&B; report ZIPs exclude weights.

## Evaluate from W&B in a fresh process

```bash
python evaluate.py \
  analysis.checkpoint=wandb://<entity>/<project>/run_<id>_model:vN \
  analysis.run_dir=/kaggle/working/traffic_sign_runs/<run-name> \
  data.root=/kaggle/input/<dataset>/za_traffic_2020 \
  data.prepared_dir=/kaggle/working/prepared_traffic_signs \
  evaluation.batch_size=1 \
  evaluation.workers=0 \
  evaluation.device=0 \
  evaluation.amp=true
```

This also works in a new Kaggle session. Analysis caches the immutable W&B model, restores its config, and checks CSV/split identity. Current CLI overrides take precedence; changing source data/split is rejected. A local checkpoint is supported with an analysis.run_dir containing the saved config and split/split.json. Older weights-only artifacts need republishing with context for this workflow.

Evaluation batch defaults to 1 independently of training. Native AP uses confidence 0.001. Operating micro-P/R/F1 use evaluation.conf=0.25 and evaluation.match_iou=0.5. Matching IoU differs from NMS IoU. Ultralytics AP and supplemental COCO AP are labeled separately.

Small-object AP uses cumulative overlapping ranges <128, <256, <512 and <1024 px² in original-image coordinates. Recall uses disjoint bins with GT counts. Empty groups are N/A. CPU predictions are reused for all cutoffs. Evaluation logs to its own W&B run linked to the source training run.

Each attempt has a unique directory; the manifest records its status and exact path. Failures raise errors and are not reported as success.

## Optional tuning, diagnostics and prediction

```bash
python tune.py \
  analysis.run_dir=/kaggle/working/traffic_sign_runs/<run-name> \
  data.root=/kaggle/input/<dataset>/za_traffic_2020 \
  data.prepared_dir=/kaggle/working/prepared_traffic_signs \
  tuning.nms_iou.metric=map50_95 \
  tuning.conf.metric=f1 \
  tuning.batch_size=1

python diagnose.py \
  analysis.run_dir=/kaggle/working/traffic_sign_runs/<run-name> \
  data.root=/kaggle/input/<dataset>/za_traffic_2020 \
  data.prepared_dir=/kaggle/working/prepared_traffic_signs \
  diagnostic.batch_size=1 \
  diagnostic.conf=0.25
```

Default tuning selects NMS IoU by validation mAP50:95, then confidence by micro-F1. Confidence sweeps filter cached CPU predictions. The tuning.run_name option remains an exact-name W&B lookup. End-to-end models reject an NMS sweep.

Pass analysis.thresholds_file=/path/to/best_thresholds.json to evaluate.py and diagnose.py for tuned results. Checkpoint checksum, split and inference geometry must match. AP still uses low confidence; operating metrics use the selected threshold. Tuning results are validation-selected, not independent test performance.

Diagnostics exports CSVs, all error overlays and up to 30 deterministic error-free samples. Green=TP, red=FP, orange=FN, purple=wrong class. A wrong class is one diagnostic event but contributes one FP plus one FN to operating metrics. Legacy diagnostic.checkpoint commands remain supported.

The notebook includes optional predict.py commands for public-test images; these have no AP metrics. Faster R-CNN remains a separate optional baseline with its existing native evaluation. The new analysis workflow supports YOLO only.

## Export and view offline

```bash
python export_reports.py \
  analysis.run_dir=/kaggle/working/traffic_sign_runs/<run-name> \
  analysis.export_dir=/kaggle/working/exports
```

Download <run-name>-reports-<export-id>.zip from Kaggle outputs, unzip and open index.html. No server, CDN, Internet, Python or source dataset is needed for viewing.

The report contains class metrics, PR/confidence curves, confusion matrices, small-object plots, optional tuning comparisons and a filterable detection gallery. Plots are also PNGs and data are CSV/JSON. Config, split manifests, W&B links and numeric training logs are included. **Training charts, weights, raw/prepared datasets, W&B caches and README files are excluded.** Missing/failed steps stay explicit. Rerun export after optional analysis; each export creates a new ZIP.

## Resume and verification

Periodic recovery artifacts retain optimizer state, last.pt, sibling best.pt and run.json. Best-model artifacts are for inference. After each successful periodic upload, older recovery versions are pruned; deletion failure is a warning.

```bash
python train.py \
  resume=<entity>/<project>/run_<id>_checkpoint:latest \
  data.root=/kaggle/input/<dataset>/za_traffic_2020 \
  data.prepared_dir=/kaggle/working/prepared_traffic_signs
```

Resume keeps the original epoch budget and optimizer. Start a new run to change the recipe. Offline runs must be synced before W&B-based analysis; retain local weights/context until upload succeeds.

Unit tests cover synthetic data, matching, provenance, artifact mocks, metrics, notebook syntax and exports. They do not prove GPU memory sufficiency, online W&B permissions or interrupted optimizer resume. Smoke-test those integrations on the target Kaggle GPU.
