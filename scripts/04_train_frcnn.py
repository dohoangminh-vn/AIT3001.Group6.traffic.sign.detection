import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.coco_eval import evaluate_coco_predictions
from src.data_utils import load_config
from src.frcnn_dataset import TrafficSignCocoDataset, detection_collate_fn, get_train_transforms
from src.frcnn_model import build_faster_rcnn


@torch.no_grad()
def export_coco_predictions(model, loader: DataLoader, device: torch.device, out_json: Path, score_thresh: float = 0.001) -> list:
    model.eval()
    prev_thresh = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = score_thresh
    preds = []
    for images, targets in loader:
        outputs = model([img.to(device) for img in images])
        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())
            for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
                sc = float(score.item())
                if sc < score_thresh:
                    continue
                x1, y1, x2, y2 = box.tolist()
                preds.append({
                    "image_id": img_id, "category_id": int(label.item()),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(sc, 5),
                })
    model.roi_heads.score_thresh = prev_thresh
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(preds), encoding="utf-8")
    return preds


def _init_wandb(cfg: dict, out_dir: Path, smoke: bool, resume_id: str | None = None):
    tr = cfg.get("tracking", {})
    mode = "disabled" if smoke else tr.get("wandb_mode", "online")
    if mode == "disabled":
        return None
    try:
        import wandb
        if mode == "online" and not os.environ.get("WANDB_API_KEY") and wandb.api.api_key is None:
            mode = "offline"
        return wandb.init(
            project=tr.get("wandb_project", "Traffic Sign"),
            entity=tr.get("wandb_entity"),
            name=out_dir.name,
            id=resume_id,
            resume="allow" if resume_id else None,
            config=cfg,
            dir=str(out_dir),
            mode=mode,
        )
    except Exception as exc:
        print(f"[W&B] Bỏ qua W&B ({exc}), vẫn log đầy đủ ra TensorBoard.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Faster R-CNN (Two-Stage)")
    parser.add_argument("--config", type=Path, default=Path("configs/frcnn_baseline.yaml"))
    parser.add_argument("--smoke", action="store_true", help="Chạy thử nhanh trên CPU")
    parser.add_argument("--default-anchors", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Đường dẫn tới last.pth để chạy tiếp")
    parser.add_argument("opts", nargs="*", help="Ghi đè cấu hình dạng key=val")
    args = parser.parse_intermixed_args()

    cfg = load_config(args.config, args.opts)
    if args.default_anchors: cfg["anchor_mode"] = "default"
    if args.epochs is not None: cfg["epochs"] = args.epochs
    if args.out_dir is not None: cfg["out_dir"] = str(args.out_dir)

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        out_dir = Path("runs/frcnn/frcnn_smoke")
        epochs = args.epochs or (cfg["epochs"] if any(o.startswith("epochs=") for o in args.opts) else 2)
        batch_size, lr, num_workers = 2, 1e-3, 0
        train_ds = TrafficSignCocoDataset(
            cfg["train_json"], cfg["img_dir"], transforms=None, limit=4,
            tiled=cfg["tiled"], copy_paste=cfg["copy_paste"], seed=cfg["seed"],
        )
        val_ds = TrafficSignCocoDataset(cfg["val_json"], cfg["img_dir"], transforms=None, limit=4)
        val_gt_path = out_dir / "_smoke_val_gt.json"
        raw_val = json.loads(Path(cfg["val_json"]).read_text(encoding="utf-8"))
        keep_ids = {im["id"] for im in val_ds.images}
        out_dir.mkdir(parents=True, exist_ok=True)
        val_gt_path.write_text(json.dumps({
            "categories": raw_val["categories"],
            "images": [im for im in raw_val["images"] if im["id"] in keep_ids],
            "annotations": [a for a in raw_val["annotations"] if a["image_id"] in keep_ids],
        }), encoding="utf-8")
        min_size, max_size, freeze_bb = (640, 640, True) if cfg["tiled"] else (400, 800, True)
    else:
        out_dir = Path(cfg["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        epochs, batch_size, lr, num_workers = cfg["epochs"], cfg["batch_size"], cfg["lr"], cfg["num_workers"]
        train_ds = TrafficSignCocoDataset(
            cfg["train_json"], cfg["img_dir"], transforms=get_train_transforms(),
            tiled=cfg["tiled"], neg_tile_ratio=cfg["neg_tile_ratio"], copy_paste=cfg["copy_paste"], seed=cfg["seed"],
        )
        val_ds = TrafficSignCocoDataset(cfg["val_json"], cfg["img_dir"], transforms=None)
        val_gt_path = Path(cfg["val_json"])
        min_size, max_size, freeze_bb = cfg["min_size"], cfg["max_size"], False

    class_counts = Counter(a["category_id"] for im in train_ds.images for a in train_ds.anns_by_img[im["id"]])
    model = build_faster_rcnn(
        num_classes=cfg["num_classes"], backbone=cfg["backbone"], anchor_mode=cfg["anchor_mode"],
        roi_size=cfg["roi_size"], loss_type=cfg["loss_type"], focal_gamma=cfg["focal_gamma"],
        use_giou=cfg["use_giou"], class_counts=dict(class_counts),
        min_size=min_size, max_size=max_size, freeze_backbone=freeze_bb,
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate_fn)

    bb_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    hd_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    param_groups = ([dict(params=bb_params, lr=lr * cfg["backbone_lr_mult"])] if bb_params else []) + [dict(params=hd_params, lr=lr)]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    sched_name = cfg["scheduler"]
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=3, min_lr=1e-6)
        if sched_name == "plateau"
        else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - cfg["warmup_epochs"]))
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_map, resume_wandb_id = 1, -1.0, None
    if args.resume and args.resume.is_file():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map", -1.0)
        resume_wandb_id = ckpt.get("wandb_id")
        print(f"Khôi phục checkpoint từ epoch {ckpt['epoch']} (best mAP@0.5:0.95={best_map:.4f})")

    try:
        cfg["_git_commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        cfg["_git_commit"] = "unknown"
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))
    wb_run = _init_wandb(cfg, out_dir, args.smoke, resume_wandb_id)

    pred_name = "frcnn_smoke_val.json" if args.smoke else f"{out_dir.name}_val.json"
    out_pred_path = Path("outputs/preds") / pred_name
    warmup_ep = 0 if args.smoke else cfg["warmup_epochs"]
    no_improve = 0

    for epoch in range(start_epoch, epochs + 1):
        if epoch <= warmup_ep:
            scale = epoch / max(1, warmup_ep)
            for g, b_lr in zip(optimizer.param_groups, base_lrs):
                g["lr"] = b_lr * scale

        model.train()
        t0, loss_sums, n_batches = time.time(), {}, 0
        for images, targets in train_loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_dict = model(images, targets)
                total_loss = sum(loss_dict.values())
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            for k, v in loss_dict.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + float(v.item())
            n_batches += 1

        avg_losses = {k: v / max(1, n_batches) for k, v in loss_sums.items()}
        avg_total = sum(avg_losses.values())

        # Đánh giá pycocotools trực tiếp trên tập Val mỗi epoch
        export_coco_predictions(model, val_loader, device, out_pred_path, cfg["eval"]["conf_export"])
        val_res = evaluate_coco_predictions(val_gt_path, out_pred_path)
        val_map = val_res["mAP@0.5:0.95"]

        if epoch > warmup_ep:
            scheduler.step(val_map) if sched_name == "plateau" else scheduler.step()

        cur_lr = optimizer.param_groups[-1]["lr"]
        metrics = {
            "train/total_loss": avg_total, **{f"train/{k}": v for k, v in avg_losses.items()},
            "train/lr": cur_lr, "val/mAP_50": val_res["mAP@0.5"],
            "val/mAP_50_95": val_map, "val/AP_small": val_res["AP_small"], "val/AR_100": val_res["AR@100"],
        }
        for k, v in metrics.items():
            writer.add_scalar(k, v, epoch)
        if wb_run is not None:
            wb_run.log(metrics, step=epoch)

        print(f"Ep {epoch:02d}/{epochs:02d} ({time.time()-t0:.1f}s) | loss={avg_total:.3f} | "
              f"mAP50={val_res['mAP@0.5']:.4f} | mAP50-95={val_map:.4f} | AP_s={val_res['AP_small']:.4f} | lr={cur_lr:.2e}")

        if val_map > best_map:
            best_map, no_improve = val_map, 0
            torch.save(model.state_dict(), out_dir / "best.pth")
        else:
            no_improve += 1

        torch.save({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best_map": best_map, "wandb_id": wb_run.id if wb_run else None,
        }, out_dir / "last.pth")

        if not args.smoke and no_improve >= cfg["patience"]:
            print(f"Early stopping tại epoch {epoch} (không tăng mAP@0.5:0.95 trong {cfg['patience']} epochs).")
            break

    writer.close()
    if wb_run is not None:
        wb_run.finish()
    print(f"Hoàn tất! Best mAP@0.5:0.95={best_map:.4f} | Weights: {(out_dir / 'best.pth').resolve()}")


if __name__ == "__main__":
    main()
