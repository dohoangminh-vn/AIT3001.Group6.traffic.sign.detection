import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import CLASS_NAMES_VI, load_config
from src.frcnn_model import build_faster_rcnn
from src.tiling import crop_tile, horizontal_tile_offsets, merge_detections


@torch.no_grad()
def run_batch4_sahi(model, img_np: np.ndarray, device: torch.device, tile_size: int, merge_method: str, score_thresh: float):
    h_img, w_img = img_np.shape[:2]
    offsets = horizontal_tile_offsets(w_img, tile_size)
    empty_b, empty_l = np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    tiles = [crop_tile(img_np, empty_b, empty_l, x0, tile_size)[0] for x0 in offsets]

    batch = [to_tensor(img_np).to(device)] + [to_tensor(t).to(device) for t in tiles]
    outputs = model(batch)
    full_out = outputs[0]

    all_b, all_s, all_l = [full_out["boxes"].cpu()], [full_out["scores"].cpu()], [full_out["labels"].cpu()]
    for x0, out in zip(offsets, outputs[1:]):
        b = out["boxes"].cpu().clone()
        if b.numel() > 0:
            b[:, [0, 2]] = (b[:, [0, 2]] + x0).clamp(0, w_img)
            b[:, [1, 3]] = b[:, [1, 3]].clamp(0, h_img)
        all_b.append(b); all_s.append(out["scores"].cpu()); all_l.append(out["labels"].cpu())

    cat_b, cat_s, cat_l = torch.cat(all_b), torch.cat(all_s), torch.cat(all_l)
    keep = cat_s >= score_thresh
    m_b, m_s, m_l = merge_detections(cat_b[keep], cat_s[keep], cat_l[keep], method=merge_method, iou_thr=0.55)
    return full_out, {"boxes": m_b, "scores": m_s, "labels": m_l}


def main():
    parser = argparse.ArgumentParser(description="Suy luận Batch-4 SAHI 640x640 cho Faster R-CNN")
    parser.add_argument("--config", type=Path, default=Path("configs/frcnn_baseline.yaml"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--val-json", type=Path, default=None)
    parser.add_argument("--img-dir", type=Path, default=None)
    parser.add_argument("--merge", choices=["wbf", "nms"], default="wbf")
    parser.add_argument("--out-pred", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("opts", nargs="*")
    args = parser.parse_intermixed_args()

    cfg = load_config(args.config, args.opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = args.weights or Path("runs/frcnn/frcnn_smoke/best.pth" if args.smoke else f"{cfg['out_dir']}/best.pth")
    val_json = args.val_json or Path(cfg["val_json"])
    img_dir = args.img_dir or Path(cfg["img_dir"])

    model = build_faster_rcnn(
        num_classes=cfg["num_classes"], backbone=cfg["backbone"], anchor_mode=cfg["anchor_mode"],
        roi_size=cfg["roi_size"], loss_type=cfg["loss_type"],
        min_size=400 if args.smoke else cfg["min_size"], max_size=800 if args.smoke else cfg["max_size"], pretrained=False,
    ).to(device)
    if weights_path.is_file():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    model.roi_heads.score_thresh = cfg["eval"]["conf_export"]

    coco_val = json.loads(val_json.read_text(encoding="utf-8"))
    images = coco_val["images"][:2] if args.smoke else coco_val["images"]
    anns_by_img = {}
    for a in coco_val["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    sahi_preds, t_full, t_sahi = [], 0.0, 0.0
    vis_sample = None

    for idx, im in enumerate(images):
        img_np = np.array(Image.open(img_dir / im["file_name"]).convert("RGB"))
        t0 = time.perf_counter()
        _ = model([to_tensor(img_np).to(device)])[0]
        t_full += time.perf_counter() - t0

        t1 = time.perf_counter()
        full_out, sahi_out = run_batch4_sahi(model, img_np, device, 640, args.merge, cfg["eval"]["conf_export"])
        t_sahi += time.perf_counter() - t1

        for b, l, s in zip(sahi_out["boxes"], sahi_out["labels"], sahi_out["scores"]):
            x1, y1, x2, y2 = b.tolist()
            sahi_preds.append({
                "image_id": im["id"], "category_id": int(l.item()),
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(float(s.item()), 5),
            })
        if vis_sample is None or len(anns_by_img.get(im["id"], [])) >= 3:
            vis_sample = (im, img_np, anns_by_img.get(im["id"], []), full_out, sahi_out)

    out_pred = args.out_pred or Path("outputs/preds") / ("frcnn_smoke_sahi_val.json" if args.smoke else f"{Path(cfg['out_dir']).name}_sahi_val.json")
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    out_pred.write_text(json.dumps(sahi_preds), encoding="utf-8")

    n = max(1, len(images))
    print(f"Full-frame: {len(images)/max(1e-6,t_full):.2f} FPS ({t_full/n*1000:.1f} ms/ảnh) | "
          f"Batch-4 SAHI ({args.merge}): {len(images)/max(1e-6,t_sahi):.2f} FPS ({t_sahi/n*1000:.1f} ms/ảnh)")
    print(f"Đã xuất {len(sahi_preds)} dự đoán SAHI -> {out_pred.resolve()}")

    if vis_sample is not None:
        im, img_np, gts, full_out, sahi_out = vis_sample
        fig, axes = plt.subplots(3, 1, figsize=(13, 11))
        for ax, title in zip(axes, [f"1. Nhãn thật ({im['file_name']})", "2. Faster R-CNN Full-Frame", f"3. Faster R-CNN Batch-4 SAHI ({args.merge.upper()})"]):
            ax.imshow(img_np); ax.set_title(title, fontweight="bold"); ax.axis("off")
        for g in gts:
            x, y, w, h = g["bbox"]
            axes[0].add_patch(patches.Rectangle((x, y), w, h, lw=2, ec="lime", fc="none"))
        for ax, out in [(axes[1], full_out), (axes[2], sahi_out)]:
            for b, l, s in zip(out["boxes"], out["labels"], out["scores"]):
                if float(s) < 0.25: continue
                x1, y1, x2, y2 = b.tolist()
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2, ec="cyan", fc="none"))
                ax.text(x1, max(0, y1 - 4), f"{CLASS_NAMES_VI.get(int(l), '')} {float(s):.2f}", color="cyan", fontsize=8, bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))
        out_vis = Path("outputs/eval/sahi_vs_full_comparison.png")
        out_vis.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(); plt.savefig(out_vis, dpi=150); plt.close()


if __name__ == "__main__":
    main()

