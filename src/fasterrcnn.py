"""Torchvision Faster R-CNN v2 training backend."""

import json
import math
from pathlib import Path

from PIL import Image

from src.data import prepare, resolve_data_root
from src.preprocess import inverse_bbox_xyxy, letterbox_geometry, online_geometry_transforms
from src.run_naming import make_run_name
from src.small_object_metrics import (
    _load_validation_ground_truth,
    evaluate_coco_predictions,
    log_small_object_metrics,
    validate_evaluation_config,
)


def _make_train_transforms(augmentation_cfg, enabled, size):
    """Build online Albumentations transforms; random operations are train-only."""
    import albumentations as A

    bbox_params = A.BboxParams(format="pascal_voc", label_fields=["class_labels"],
                               min_visibility=0.0, clip=True)
    transforms = online_geometry_transforms(size)
    if enabled:
        color = augmentation_cfg.color_jitter
        transforms.append(A.ColorJitter(
            brightness=float(color.brightness), contrast=float(color.contrast),
            saturation=float(color.saturation), hue=float(color.hue), p=0.5,
        ))
        flip_probability = float(augmentation_cfg.horizontal_flip_probability)
        if not 0.0 <= flip_probability <= 1.0:
            raise ValueError("model.augmentation.horizontal_flip_probability must be in [0, 1]")
        transforms.append(A.HorizontalFlip(p=flip_probability))
        affine = augmentation_cfg.affine
        translate = tuple(float(value) for value in affine.translate)
        scale = tuple(float(value) for value in affine.scale)
        if len(translate) != 2 or len(scale) != 2:
            raise ValueError("Faster R-CNN affine translate/scale must have two values")
        transforms.append(A.Affine(
            rotate=(-float(affine.degrees), float(affine.degrees)),
            translate_percent={"x": (-translate[0], translate[0]), "y": (-translate[1], translate[1])},
            scale=(scale[0], scale[1]), p=0.5,
        ))
    return A.Compose(transforms, bbox_params=bbox_params)


class TrafficSignDataset:
    """Online dataset adapter over raw images and original-coordinate boxes."""

    def __init__(self, data_root, subset, records, size=640, transforms=None):
        import torch
        from torchvision import tv_tensors
        from torchvision.transforms.functional import to_tensor

        self.torch = torch
        self.to_tensor = to_tensor
        self.tv_tensors = tv_tensors
        self.transforms = transforms
        self.root = resolve_data_root(data_root) / "traffic_train" / "images"
        self.subset = subset
        self.size = int(size)
        self.records = [record for record in records.values() if record["subset"] == subset]
        self.records.sort(key=lambda record: record["image_id"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = self.root / record["file_name"]
        with Image.open(image_path) as opened:
            opened = opened.convert("RGB")
            original_width, original_height = opened.size
            image = __import__("numpy").array(opened)
        boxes_xyxy = []
        labels_list = []
        for class_id, cx, cy, bw, bh in record["labels"]:
            boxes_xyxy.append([(cx - bw / 2) * original_width, (cy - bh / 2) * original_height,
                               (cx + bw / 2) * original_width, (cy + bh / 2) * original_height])
            labels_list.append(int(class_id) + 1)
        transformed = self.transforms(image=image, bboxes=boxes_xyxy, class_labels=labels_list)
        image = self.to_tensor(transformed["image"])
        rows = [(*box, label) for box, label in zip(transformed["bboxes"], transformed["class_labels"])]
        if rows:
            boxes = self.torch.tensor([row[:4] for row in rows], dtype=self.torch.float32)
            labels = self.torch.tensor([row[4] for row in rows], dtype=self.torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            boxes = self.torch.zeros((0, 4), dtype=self.torch.float32)
            labels = self.torch.zeros((0,), dtype=self.torch.int64)
            area = self.torch.zeros((0,), dtype=self.torch.float32)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": self.torch.tensor(record["image_id"], dtype=self.torch.int64),
            "area": area,
            "iscrowd": self.torch.zeros((len(rows),), dtype=self.torch.int64),
        }
        target["area"] = ((target["boxes"][:, 2] - target["boxes"][:, 0])
                          * (target["boxes"][:, 3] - target["boxes"][:, 1]))
        return image, target


def _collate(batch):
    return tuple(zip(*batch))


def _build_model(size=640, weights_name="COCO_V1", inference_cfg=None):
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    try:
        weights = getattr(FasterRCNN_ResNet50_FPN_V2_Weights, str(weights_name))
    except AttributeError as error:
        raise ValueError(f"Unsupported Faster R-CNN v2 weights: {weights_name}") from error
    inference_cfg = inference_cfg or {}
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        min_size=size,
        max_size=size,
        box_score_thresh=float(inference_cfg.get("score_threshold", 0.001)),
        box_nms_thresh=float(inference_cfg.get("nms_iou", 0.7)),
        box_detections_per_img=int(inference_cfg.get("max_detections", 300)),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=8)
    return model


def _predict(model, loader, dataset, device):
    import torch

    model.eval()
    predictions = []
    with torch.inference_mode():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"])
                record = dataset.record_by_id[image_id]
                original = dataset.original_by_id[image_id]
                boxes = output["boxes"].detach().cpu().tolist()
                scores = output["scores"].detach().cpu().tolist()
                labels = output["labels"].detach().cpu().tolist()
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = inverse_bbox_xyxy(
                        box, record["geometry"], original["width"], original["height"]
                    )
                    if x2 <= x1 or y2 <= y1:
                        continue
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(label) - 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    })
    return predictions


def train(cfg):
    import torch
    import wandb
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    if cfg.resume:
        raise ValueError("Faster R-CNN resume is not implemented yet; start a new run")
    validate_evaluation_config(
        cfg.evaluation.inference, cfg.evaluation.small_object_area_cutoffs_px2
    )
    prepare(cfg.data.root, cfg.data.prepared_dir, cfg.seed, cfg.data.val_fraction)
    from src.data import read_annotations
    image_records, _ = read_annotations(cfg.data.root)
    split_manifest = json.loads((Path(cfg.data.prepared_dir) / "split.json").read_text(encoding="utf-8"))
    subsets = {int(image_id): subset for subset, ids in split_manifest["splits"].items()
               for image_id in ids}
    records = {image_id: {**record, "image_id": image_id, "subset": subsets[image_id],
                          "geometry": letterbox_geometry(record["width"], record["height"],
                                                         int(cfg.model.imgsz))}
               for image_id, record in image_records.items() if image_id in subsets}
    _, image_map = _load_validation_ground_truth(cfg.data.root, cfg.data.prepared_dir)

    train_transforms = _make_train_transforms(
        cfg.model.augmentation, bool(cfg.augmentation.enabled), int(cfg.model.imgsz)
    )
    train_dataset = TrafficSignDataset(
        cfg.data.root, "train", records, cfg.model.imgsz, transforms=train_transforms
    )
    val_dataset = TrafficSignDataset(
        cfg.data.root, "val", records, cfg.model.imgsz,
        transforms=_make_train_transforms(cfg.model.augmentation, False, int(cfg.model.imgsz)),
    )
    val_dataset.record_by_id = {record["image_id"]: record for record in val_dataset.records}
    val_dataset.original_by_id = image_map
    train_loader = DataLoader(
        train_dataset, batch_size=int(cfg.model.batch_size), shuffle=True,
        num_workers=int(cfg.model.workers), collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(cfg.model.batch_size), shuffle=False,
        num_workers=int(cfg.model.workers), collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
    )
    if str(cfg.model.device) == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cfg.model.device}")
    use_amp = bool(cfg.model.amp and device.type == "cuda")
    import random

    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model = _build_model(
        int(cfg.model.imgsz), cfg.model.weights,
        {key: cfg.evaluation.inference[key] for key in cfg.evaluation.inference},
    ).to(device)
    optimizer_cfg = cfg.model.optimizer
    if str(optimizer_cfg.name).lower() != "adamw":
        raise ValueError("The Faster R-CNN baseline currently supports optimizer.name=AdamW only")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(optimizer_cfg.lr), weight_decay=float(optimizer_cfg.weight_decay),
    )
    scheduler_cfg = cfg.model.scheduler
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(scheduler_cfg.warmup_start_factor),
        end_factor=1.0,
        total_iters=int(scheduler_cfg.warmup_epochs),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(scheduler_cfg.cosine_epochs), eta_min=float(scheduler_cfg.eta_min),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[int(scheduler_cfg.warmup_epochs)],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if int(cfg.tracking.checkpoint_every) < 1:
        raise ValueError("tracking.checkpoint_every must be positive")

    run_name = make_run_name(cfg.tracking.name, default_prefix="fasterrcnn-r50fpn-v2")
    run_dir = Path(cfg.output_dir).resolve() / run_name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=False)
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    (run_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    dataset_gt, _ = _load_validation_ground_truth(cfg.data.root, cfg.data.prepared_dir)
    best_ap, best_epoch, stale_epochs = -math.inf, 0, 0

    with wandb.init(
        project=cfg.tracking.project,
        entity=cfg.tracking.entity,
        mode=cfg.tracking.mode,
        name=run_name,
        dir=str(run_dir),
        config={
            "model": f"FasterRCNN_ResNet50_FPN_V2_Weights.{cfg.model.weights}",
            "pretrained": True,
            "input_size": int(cfg.model.imgsz),
            "batch_size": int(cfg.model.batch_size),
            "epochs": int(cfg.model.epochs),
            "patience": int(cfg.model.patience),
            "device": str(device),
            "amp": use_amp,
            "workers": int(cfg.model.workers),
            "optimizer": "AdamW",
            "lr": float(optimizer_cfg.lr),
            "weight_decay": float(optimizer_cfg.weight_decay),
            "scheduler": "LinearLR+CosineAnnealingLR",
            "scheduler_config": {
                "warmup_epochs": int(scheduler_cfg.warmup_epochs),
                "warmup_start_factor": float(scheduler_cfg.warmup_start_factor),
                "cosine_epochs": int(scheduler_cfg.cosine_epochs),
                "eta_min": float(scheduler_cfg.eta_min),
            },
            "inference": {key: cfg.evaluation.inference[key] for key in cfg.evaluation.inference},
            "augmentation_enabled": bool(cfg.augmentation.enabled),
            "resolved_config": resolved_config,
            "seed": int(cfg.seed),
            "torch_version": torch.__version__,
            "torchvision_version": __import__("torchvision").__version__,
            "preprocess": {"online": True, "library": "albumentations",
                           "size": int(cfg.model.imgsz), "letterbox": "centered"},
        },
    ) as run:
        for epoch in range(int(cfg.model.epochs)):
            model.train()
            loss_sum, steps = 0.0, 0
            for images, targets in train_loader:
                images = [image.to(device) for image in images]
                targets = [{key: value.to(device) for key, value in target.items()}
                           for target in targets]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite Faster R-CNN loss at epoch {epoch + 1}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach().cpu())
                steps += 1

            scheduler.step()
            predictions = _predict(model, val_loader, val_dataset, device)
            ap50, ap50_95 = evaluate_coco_predictions(dataset_gt, predictions)
            mean_loss = loss_sum / max(steps, 1)
            run.log({
                "epoch": epoch + 1,
                "train/loss": mean_loss,
                "val/AP50": ap50,
                "val/AP50_95": ap50_95,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch + 1)

            improved = ap50_95 > best_ap
            if improved:
                best_ap, best_epoch, stale_epochs = ap50_95, epoch + 1, 0
                torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                            "best_ap50_95": best_ap}, weights_dir / "best.pt")
            else:
                stale_epochs += 1
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "epoch": epoch + 1, "best_ap50_95": best_ap}, weights_dir / "last.pt")
            print(f"Epoch {epoch + 1}/{cfg.model.epochs}: loss={mean_loss:.4f}, "
                  f"AP50={ap50:.4f}, AP50:95={ap50_95:.4f}")

            if (epoch + 1) % int(cfg.tracking.checkpoint_every) == 0 or improved:
                artifact = wandb.Artifact(f"{run.id}-checkpoint", type="checkpoint",
                                          metadata={"epoch": epoch + 1, "best_ap50_95": best_ap})
                artifact.add_file(str(weights_dir / "last.pt"), name="last.pt")
                if (weights_dir / "best.pt").is_file():
                    artifact.add_file(str(weights_dir / "best.pt"), name="best.pt")
                run.log_artifact(artifact, aliases=["latest"])
            if stale_epochs >= int(cfg.model.patience):
                break

        best_checkpoint = weights_dir / "best.pt"
        checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        predictions = _predict(model, val_loader, val_dataset, device)
        ap50, ap50_95 = evaluate_coco_predictions(dataset_gt, predictions)
        run.log({"best/epoch": best_epoch, "best/AP50": ap50, "best/AP50_95": ap50_95})
        log_small_object_metrics(
            dataset_gt, predictions, run_dir / "small_object_metrics.csv", run,
            area_cutoffs=cfg.evaluation.small_object_area_cutoffs_px2,
        )
        artifact = wandb.Artifact(f"{run.id}-model", type="model",
                                  metadata={"best_epoch": best_epoch, "AP50_95": ap50_95})
        artifact.add_file(str(best_checkpoint), name="best.pt")
        run.log_artifact(artifact, aliases=["best"])
    print(f"Best Faster R-CNN checkpoint: {weights_dir / 'best.pt'}")
