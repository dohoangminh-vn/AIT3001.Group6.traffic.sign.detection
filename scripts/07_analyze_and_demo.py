import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import CLASS_NAMES_ASCII, load_config
from src.frcnn_model import build_faster_rcnn


def run_feature_gradcam(model, img_path: Path, device: torch.device, out_img: Path) -> None:
    """Trực quan hóa FPN Feature Maps (P2..P6) và RoI Grad-CAM tại lớp conv cuối của box_head."""
    img_np = np.array(Image.open(img_path).convert("RGB"))
    img_t = to_tensor(img_np).to(device)
    fpn_feats, roi_act, roi_grad = {}, {}, {}

    h_fpn = model.backbone.fpn.register_forward_hook(lambda _m, _i, out: fpn_feats.update({k: v.detach().cpu() for k, v in out.items()}))
    conv_target = next((m for m in reversed(list(model.roi_heads.box_head.modules())) if isinstance(m, torch.nn.Conv2d)), None)
    h_act = conv_target.register_forward_hook(lambda _m, _i, out: roi_act.update({"feat": out})) if conv_target else None
    h_grd = conv_target.register_full_backward_hook(lambda _m, _gi, go: roi_grad.update({"grad": go[0]})) if conv_target else None

    model.eval()
    img_list, _ = model.transform([img_t], None)
    feats = model.backbone(img_list.tensors)
    props, _ = model.rpn(img_list, feats, None)
    box_feats = model.roi_heads.box_head(model.roi_heads.box_roi_pool(feats, props, img_list.image_sizes))
    cls_logits, box_reg = model.roi_heads.box_predictor(box_feats)

    # Grad-CAM cho RoI dự đoán biển báo tự tin nhất
    fg_scores = torch.softmax(cls_logits, dim=-1)[:, 1:]
    best_roi, best_cls_rel = divmod(int(torch.argmax(fg_scores).item()), fg_scores.shape[1])
    model.zero_grad()
    cls_logits[best_roi, best_cls_rel + 1].backward()

    h_fpn.remove()
    if h_act: h_act.remove()
    if h_grd: h_grd.remove()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes[0, 0].imshow(img_np); axes[0, 0].set_title(f"Ảnh gốc ({img_path.name})", fontweight="bold")
    for ax, (lvl, title) in zip([axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]], [("0", "FPN P2 (stride 4)"), ("1", "FPN P3 (stride 8)"), ("2", "FPN P4 (stride 16)"), ("3", "FPN P5 (stride 32)")]):
        if lvl in fpn_feats:
            ax.imshow(fpn_feats[lvl][0].abs().mean(dim=0).numpy(), cmap="magma")
        ax.set_title(title, fontweight="bold")

    # Vẽ RoI Grad-CAM phóng đại
    if "feat" in roi_act and "grad" in roi_grad:
        act = roi_act["feat"][best_roi].detach().cpu()
        w = roi_grad["grad"][best_roi].detach().cpu().mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((w * act).sum(dim=0)).numpy()
        cam = (cam - cam.min()) / max(1e-6, cam.max() - cam.min())
        sx, sy = img_np.shape[1] / img_list.image_sizes[0][1], img_np.shape[0] / img_list.image_sizes[0][0]
        x1, y1, x2, y2 = props[0][best_roi].detach().cpu().numpy()
        x1, x2 = max(0, int(x1 * sx)), min(img_np.shape[1], max(int(x1 * sx) + 4, int(x2 * sx)))
        y1, y2 = max(0, int(y1 * sy)), min(img_np.shape[0], max(int(y1 * sy) + 4, int(y2 * sy)))
        crop = img_np[y1:y2, x1:x2]
        cam_rs = cv2.resize(cam, (crop.shape[1], crop.shape[0]))
        axes[1, 2].imshow(crop); axes[1, 2].imshow(cam_rs, cmap="jet", alpha=0.45)
        axes[1, 2].set_title(f"RoI Grad-CAM ({cam.shape[0]}×{cam.shape[1]} | {CLASS_NAMES_ASCII[best_cls_rel]})", fontweight="bold")
    for ax in axes.flat:
        ax.axis("off")
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_img, dpi=180); plt.close()
    print(f"Đã lưu FPN + RoI Grad-CAM -> {out_img.resolve()}")


@torch.no_grad()
def run_video_demo(model, frame_paths: list[Path], device: torch.device, out_mp4: Path, conf: float = 0.35) -> None:
    model.eval()
    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (w, h))
    use_amp = device.type == "cuda"
    t_total = 0.0

    for p in frame_paths:
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model([to_tensor(rgb).to(device)])[0]
        dt = time.perf_counter() - t0
        t_total += dt
        for box, lbl, sc in zip(out["boxes"], out["labels"], out["scores"]):
            if float(sc) < conf: continue
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 180), 2)
            txt = f"{CLASS_NAMES_ASCII.get(int(lbl)-1, int(lbl))} {float(sc):.2f}"
            cv2.putText(bgr, txt, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(bgr, f"Faster R-CNN | {1.0/max(1e-6,dt):.1f} FPS ({dt*1000:.0f} ms)", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)
        writer.write(bgr)

    writer.release()
    print(f"Video demo ({len(frame_paths)} frames, avg {len(frame_paths)/max(1e-6,t_total):.2f} FPS) -> {out_mp4.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa FPN/Grad-CAM & Video Demo FPS")
    parser.add_argument("--mode", choices=["features", "video"], required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/frcnn_baseline.yaml"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("opts", nargs="*")
    args = parser.parse_intermixed_args()

    cfg = load_config(args.config, args.opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = args.weights or Path("runs/frcnn/frcnn_smoke/best.pth" if args.smoke else f"{cfg['out_dir']}/best.pth")

    model = build_faster_rcnn(
        num_classes=cfg["num_classes"], backbone=cfg["backbone"], anchor_mode=cfg["anchor_mode"],
        roi_size=cfg["roi_size"], loss_type=cfg["loss_type"],
        min_size=400 if args.smoke else cfg["min_size"], max_size=800 if args.smoke else cfg["max_size"], pretrained=False,
    ).to(device)
    if weights_path.is_file():
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))

    val_coco = json.loads(Path(cfg["val_json"]).read_text(encoding="utf-8"))
    img_dir = Path(cfg["img_dir"])
    if args.mode == "features":
        best_im = max(val_coco["images"], key=lambda im: sum(1 for a in val_coco["annotations"] if a["image_id"] == im["id"]))
        run_feature_gradcam(model, img_dir / best_im["file_name"], device, Path("outputs/eval/frcnn_fpn_and_gradcam.png"))
    else:
        top_street = Counter(im["street_id"] for im in val_coco["images"]).most_common(1)[0][0]
        frames = sorted([img_dir / im["file_name"] for im in val_coco["images"] if im["street_id"] == top_street], key=lambda p: int(p.stem))
        run_video_demo(model, frames[:3] if args.smoke else frames[:60], device, Path("outputs/eval/demo_dashcam.mp4"))


if __name__ == "__main__":
    main()

