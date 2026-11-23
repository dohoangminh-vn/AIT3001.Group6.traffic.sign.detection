import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import CLASS_NAMES_VI
from src.frcnn_dataset import (
    TrafficSignCocoDataset,
    detection_collate_fn,
    get_train_transforms,
)
from src.frcnn_model import build_faster_rcnn


@torch.no_grad()
def export_coco_predictions(
    model, loader: DataLoader, device: torch.device, out_json: Path, score_thresh: float = 0.001
) -> list:
    """Chạy suy luận trên tập val và xuất COCO JSON chung định dạng với YOLO."""
    model.eval()
    model.roi_heads.score_thresh = score_thresh
    preds = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())
            for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
                sc = float(score.item())
                if sc < score_thresh:
                    continue
                x1, y1, x2, y2 = box.tolist()
                preds.append({
                    "image_id": img_id,
                    "category_id": int(label.item()),  # Đã là 1..7 chuẩn COCO
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(sc, 5),
                })

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(preds), encoding="utf-8")
    print(f"Đã xuất {len(preds)} dự đoán Faster R-CNN chuẩn COCO -> {out_json.resolve()}")
    return preds


@torch.no_grad()
def save_visual_sample(model, dataset: TrafficSignCocoDataset, device: torch.device, out_img: Path):
    """Vẽ 1 ảnh so sánh Nhãn thật (xanh lá) và Dự đoán Faster R-CNN (đỏ) để thấy kết quả ngay."""
    model.eval()
    img_t, tgt = dataset[0]
    pred = model([img_t.to(device)])[0]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(img_t.permute(1, 2, 0).cpu().numpy())

    # Vẽ nhãn thật (xanh lá)
    for box, lbl in zip(tgt["boxes"], tgt["labels"]):
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2, ec="lime", fc="none"))
        ax.text(x1, max(0, y1 - 5), f"GT: {CLASS_NAMES_VI[int(lbl)]}", color="lime", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))

    # Vẽ dự đoán (đỏ, lấy tối đa 10 box có score cao nhất > 0.2)
    shown = 0
    for box, lbl, sc in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if sc.item() < 0.2 or shown >= 10:
            continue
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2, ec="red", ls="--", fc="none"))
        ax.text(x1, y2 + 12, f"Pred: {CLASS_NAMES_VI.get(int(lbl), int(lbl))} ({sc.item():.2f})",
                color="yellow", fontsize=8, bbox=dict(facecolor="red", alpha=0.6, pad=1, edgecolor="none"))
        shown += 1

    ax.set_title(f"Faster R-CNN Smoke Check — Xanh: Nhãn thật | Đỏ nét đứt: Dự đoán ({shown} box > 0.2)")
    ax.axis("off")
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    plt.close()
    print(f"Đã lưu ảnh kiểm tra trực quan -> {out_img.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Faster R-CNN (Two-Stage)")
    parser.add_argument("--config", type=Path, default=Path("configs/frcnn_baseline.yaml"))
    parser.add_argument("--smoke", action="store_true", help="Chạy thử nhanh 4 ảnh trên CPU")
    parser.add_argument("--default-anchors", action="store_true", help="Dùng anchor mặc định (32..512) cho Ablation")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.default_anchors:
        cfg["small_anchors"] = False
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.out_dir is not None:
        cfg["out_dir"] = str(args.out_dir)

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        print("=== CHẾ ĐỘ SMOKE TEST FASTER R-CNN (4 ảnh, 5 epochs, CPU) ===")
        out_dir = Path("runs/frcnn/frcnn_smoke")
        epochs = 5
        batch_size = 2
        lr = 1e-3
        train_ds = TrafficSignCocoDataset(cfg["train_json"], cfg["img_dir"], transforms=None, limit=4)
        val_ds = TrafficSignCocoDataset(cfg["val_json"], cfg["img_dir"], transforms=None, limit=8)
        model = build_faster_rcnn(
            num_classes=cfg["num_classes"],
            small_anchors=cfg["small_anchors"],
            min_size=400,
            max_size=800,
            freeze_backbone=True,  # Đóng băng backbone để chạy nhanh trên CPU
        ).to(device)
        num_workers = 0
    else:
        print(f"=== CHẾ ĐỘ FULL TRAIN FASTER R-CNN (small_anchors={cfg['small_anchors']} | device={device}) ===")
        out_dir = Path(cfg["out_dir"])
        epochs = cfg["epochs"]
        batch_size = cfg["batch_size"]
        lr = cfg["lr"]
        train_ds = TrafficSignCocoDataset(
            cfg["train_json"], cfg["img_dir"], transforms=get_train_transforms()
        )
        val_ds = TrafficSignCocoDataset(cfg["val_json"], cfg["img_dir"], transforms=None)
        model = build_faster_rcnn(
            num_classes=cfg["num_classes"],
            small_anchors=cfg["small_anchors"],
            min_size=cfg["min_size"],
            max_size=cfg["max_size"],
            freeze_backbone=False,
        ).to(device)
        num_workers = cfg["num_workers"]

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=detection_collate_fn,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("Anchor sizes đang dùng:", model.rpn.anchor_generator.sizes)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        loss_sums = {}
        n_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_dict = model(images, targets)
                total_loss = sum(loss_dict.values())

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for k, v in loss_dict.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + float(v.item())
            n_batches += 1

        scheduler.step()
        avg_losses = {k: v / max(1, n_batches) for k, v in loss_sums.items()}
        avg_total = sum(avg_losses.values())

        # Ghi log TensorBoard (bắt buộc chia sẻ với giảng viên!)
        writer.add_scalar("train/total_loss", avg_total, epoch)
        for k, v in avg_losses.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        parts = " | ".join(f"{k}={v:.3f}" for k, v in avg_losses.items())
        print(f"Epoch {epoch:02d}/{epochs:02d} ({time.time() - t0:.1f}s) | total={avg_total:.3f} | {parts}")

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(model.state_dict(), out_dir / "best.pth")

    writer.close()
    print(f"\nĐã lưu trọng số tốt nhất tại: {(out_dir / 'best.pth').resolve()}")

    # Vẽ 1 ảnh kiểm tra trực quan (trên ảnh train[0] nếu smoke để thấy model đã bắt đầu học)
    save_visual_sample(model, train_ds if args.smoke else val_ds, device, Path("outputs/ch4_frcnn_smoke_pred.jpg"))

    # Xuất dự đoán chuẩn COCO trên tập val
    pred_name = "frcnn_smoke_val.json" if args.smoke else f"{out_dir.name}_val.json"
    export_coco_predictions(model, val_loader, device, Path("outputs/preds") / pred_name)


if __name__ == "__main__":
    main()