import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import audit_and_clean_coco, compute_iou_xywh, load_config


def _pairwise_intra_ious(anns_list: list) -> tuple[list[float], list[float]]:
    all_ious, max_per_img = [], []
    by_img = defaultdict(list)
    for a in anns_list:
        by_img[a["image_id"]].append(a["bbox"])
    for boxes in by_img.values():
        if len(boxes) < 2:
            continue
        img_ious = [
            compute_iou_xywh(boxes[i], boxes[j])
            for i in range(len(boxes))
            for j in range(i + 1, len(boxes))
        ]
        all_ious.extend(img_ious)
        max_per_img.append(max(img_ious))
    return all_ious, max_per_img


def main():
    cfg = load_config(Path("configs/base.yaml"), sys.argv[1:])
    data_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path("outputs/eda")
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = json.loads((data_dir / "train_traffic_sign_dataset.json").read_text(encoding="utf-8"))
    images, anns = coco["images"], coco["annotations"]
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {im["id"]: im for im in images}

    widths = np.array([a["bbox"][2] for a in anns], dtype=float)
    heights = np.array([a["bbox"][3] for a in anns], dtype=float)
    areas, scales = widths * heights, np.sqrt(widths * heights)
    ars = heights / np.maximum(widths, 1e-6)

    n_small = int((areas < 32**2).sum())
    n_med = int(((areas >= 32**2) & (areas < 96**2)).sum())
    n_large = int((areas >= 96**2).sum())

    # Thống kê số box mỗi ảnh & IoU nội ảnh trước/sau làm sạch
    boxes_per_img = np.array([sum(1 for a in anns if a["image_id"] == im["id"]) for im in images], dtype=float)
    cleaned_coco, audit_rep = audit_and_clean_coco(
        coco, cfg["data"]["dedup_iou"], cfg["data"]["conflict_iou"], cfg["data"]["min_box_side"]
    )
    raw_ious, raw_max_ious = _pairwise_intra_ious(anns)
    clean_ious, clean_max_ious = _pairwise_intra_ious(cleaned_coco["annotations"])

    cat_counts = Counter(a["category_id"] for a in anns)
    clean_cat_counts = Counter(a["category_id"] for a in cleaned_coco["annotations"])
    street_counts = Counter(im.get("street_id", -1) for im in images)

    print("=" * 64)
    print(f"Ảnh: {len(images)} | Box gốc: {len(anns)} | Box sạch: {len(cleaned_coco['annotations'])} | Lớp: {len(cats)}")
    print(f"Box/ảnh: min={boxes_per_img.min():.0f}, mean={boxes_per_img.mean():.2f}, median={np.median(boxes_per_img):.0f}, std={boxes_per_img.std():.2f}, max={boxes_per_img.max():.0f}")
    print(f"IoU nội ảnh (gốc):   min={min(raw_ious):.3f}, mean={np.mean(raw_ious):.4f}, max={max(raw_ious):.3f} | mean(max/ảnh)={np.mean(raw_max_ious):.4f}")
    print(f"IoU nội ảnh (sạch):  min={min(clean_ious):.3f}, mean={np.mean(clean_ious):.4f}, max={max(clean_ious):.3f}")
    print(f"Làm sạch: cùng lớp={audit_rep['removed_same_cls']} | xung đột khác lớp={audit_rep['removed_cross_cls']} | suy biến={audit_rep['removed_degenerate']}")
    print(f"Small (<32^2): {n_small} ({n_small/len(anns)*100:.1f}%) | Median w×h: {np.median(widths):.1f}×{np.median(heights):.1f} | AR(h/w) p5-p95: [{np.percentile(ars, 5):.2f}, {np.percentile(ars, 95):.2f}]")
    print("=" * 64)

    summary = {
        "num_images": len(images),
        "num_annotations_raw": len(anns),
        "num_annotations_clean": len(cleaned_coco["annotations"]),
        "num_streets": len(street_counts),
        "coco_scale": {"small": n_small, "medium": n_med, "large": n_large},
        "median_wh": [float(np.median(widths)), float(np.median(heights))],
        "aspect_ratio_p05_p50_p95": [round(float(np.percentile(ars, p)), 3) for p in (5, 50, 95)],
        "scale_p10_p50_p90_p95": [round(float(np.percentile(scales, p)), 1) for p in (10, 50, 90, 95)],
        "boxes_per_image": {
            "min": int(boxes_per_img.min()), "mean": round(float(boxes_per_img.mean()), 3),
            "median": float(np.median(boxes_per_img)), "std": round(float(boxes_per_img.std()), 3), "max": int(boxes_per_img.max()),
        },
        "intra_image_iou": {
            "raw_min": round(float(min(raw_ious)), 4), "raw_mean": round(float(np.mean(raw_ious)), 4),
            "raw_max": round(float(max(raw_ious)), 4), "raw_mean_max_per_img": round(float(np.mean(raw_max_ious)), 4),
            "clean_mean": round(float(np.mean(clean_ious)), 4), "clean_max": round(float(max(clean_ious)), 4),
        },
        "cleaning_audit": {k: v for k, v in audit_rep.items() if k != "conflicts"},
        "class_counts_raw": {cats[cid]: cat_counts[cid] for cid in sorted(cats)},
        "class_counts_clean": {cats[cid]: clean_cat_counts[cid] for cid in sorted(cats)},
    }
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Biểu đồ 1: Phân bố 7 lớp
    sorted_cids = sorted(cats, key=lambda c: cat_counts[c], reverse=True)
    labels = [f"[{c}] {cats[c]}" for c in sorted_cids]
    values = [cat_counts[c] for c in sorted_cids]
    colors = ["#2b7bba" if v > 1000 else "#d95f02" for v in values]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + 35, bar.get_y() + bar.get_height() / 2, f"{val} ({val/len(anns)*100:.1f}%)", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("Số lượng bounding box")
    ax.set_title("Phân bố 7 lớp biển báo (màu cam: lớp hiếm < 1,000 mẫu)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(out_dir / "01_class_distribution.png", dpi=200); plt.close()

    # Biểu đồ 2: Kích thước bbox
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(widths, heights, s=6, alpha=0.25, color="#1f77b4")
    axes[0].axvline(32, color="red", ls="--", label="Ngưỡng Small (32 px)"); axes[0].axhline(32, color="red", ls="--")
    axes[0].set(xlim=(0, 150), ylim=(0, 150), xlabel="Chiều rộng box (px)", ylabel="Chiều cao box (px)", title="Kích thước box gốc (zoom 0..150 px)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    scale_vals = [n_small, n_med, n_large]
    bars2 = axes[1].bar(["Small\n(< 32×32)", "Medium\n(32²..96²)", "Large\n(≥ 96×96)"], scale_vals, color=["#e74c3c", "#f39c12", "#27ae60"])
    for bar, val in zip(bars2, scale_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 150, f"{val}\n({val/len(anns)*100:.1f}%)", ha="center", fontweight="bold")
    axes[1].set(ylim=(0, max(scale_vals) * 1.18), ylabel="Số lượng biển báo", title="Tỉ lệ vật thể theo chuẩn COCO")
    axes[1].grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(out_dir / "02_bbox_scale_analysis.png", dpi=200); plt.close()

    # Biểu đồ 3: street_id + mẫu phóng đại
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 1])
    ax_top = fig.add_subplot(gs[0, :])
    top_streets = street_counts.most_common()
    ax_top.bar([str(k) for k, _ in top_streets], [v for _, v in top_streets], color="#4c72b0")
    ax_top.set(title=f"Số khung hình theo {len(street_counts)} street_id — Lý do PHẢI chia theo nhóm!", xlabel="street_id", ylabel="Số ảnh")
    ax_top.tick_params(axis="x", rotation=60, labelsize=8); ax_top.grid(axis="y", ls="--", alpha=0.4)
    sorted_anns = sorted(anns, key=lambda a: a["bbox"][2] * a["bbox"][3])
    for idx, ann_i in enumerate([100, 1500, 4500, 8000, 10200, 10950]):
        a = sorted_anns[ann_i]
        im = Image.open(data_dir / "images" / img_map[a["image_id"]]["file_name"]).convert("RGB")
        x, y, w, h = [int(v) for v in a["bbox"]]
        pad = max(6, int(max(w, h) * 0.35))
        ax_sub = fig.add_subplot(gs[1, idx])
        ax_sub.imshow(im.crop((max(0, x - pad), max(0, y - pad), min(im.width, x + w + pad), min(im.height, y + h + pad))))
        ax_sub.set_title(f"{cats[a['category_id']]}\n{w}×{h} px", fontsize=9); ax_sub.axis("off")
    plt.tight_layout(); plt.savefig(out_dir / "03_street_and_samples.png", dpi=200); plt.close()

    # Biểu đồ 4: Hình học Anchor (AR), Spatial Heatmap (3-Tile) & Kiểm toán IoU/Xung đột
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].hist(ars[(ars >= 0.3) & (ars <= 2.2)], bins=45, color="#3b75af", edgecolor="white")
    axes[0, 0].axvspan(0.74, 1.28, color="#2ecc71", alpha=0.2, label="90% biển báo [0.74, 1.28]")
    for ar_val, col, lbl in [(0.75, "green", "Sign AR (0.75, 1.0, 1.3)"), (1.0, "green", None), (1.3, "green", None), (0.5, "red", "COCO AR (0.5, 2.0)"), (2.0, "red", None)]:
        axes[0, 0].axvline(ar_val, color=col, ls="--", lw=1.4, label=lbl)
    axes[0, 0].set(title="Tỷ lệ khung hình h/w (minh chứng Anchor mới)", xlabel="h / w", ylabel="Số lượng")
    axes[0, 0].legend(fontsize=8)

    xc = np.array([a["bbox"][0] + a["bbox"][2] / 2 for a in cleaned_coco["annotations"]])
    yc = np.array([a["bbox"][1] + a["bbox"][3] / 2 for a in cleaned_coco["annotations"]])
    axes[0, 1].hist2d(xc, yc, bins=[40, 18], range=[[0, 1622], [0, 626]], cmap="inferno")
    for x_line in [491, 640, 982, 1131]:
        axes[0, 1].axvline(x_line, color="cyan", ls="--", lw=1.3)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set(title="Bản đồ vị trí biển báo & 3 lát ngang 640 (ov 149 px)", xlabel="x (px)", ylabel="y (px)")

    axes[1, 0].hist([raw_ious, clean_ious], bins=25, label=[f"Gốc (max={max(raw_ious):.2f})", f"Sạch (max={max(clean_ious):.2f})"], color=["#e74c3c", "#27ae60"])
    axes[1, 0].set(yscale="log", title="Phân phối IoU nội ảnh trước & sau làm sạch", xlabel="Pairwise IoU", ylabel="Số cặp box (log)")
    axes[1, 0].legend(fontsize=8)

    if audit_rep["conflicts"]:
        c0 = audit_rep["conflicts"][-1]
        im_c = Image.open(data_dir / "images" / img_map[c0["image_id"]]["file_name"]).convert("RGB")
        bx, by, bw, bh = [int(v) for v in c0["bbox"]]
        pad = 32
        axes[1, 1].imshow(im_c.crop((max(0, bx - pad), max(0, by - pad), min(im_c.width, bx + bw + pad), min(im_c.height, by + bh + pad))))
        axes[1, 1].add_patch(patches.Rectangle((min(pad, bx), min(pad, by)), bw, bh, lw=2, ec="yellow", fc="none"))
        axes[1, 1].set_title(f"Lỗi nhãn đè khác lớp (ảnh {c0['image_id']}): [{c0['kept_cat']}] vs [{c0['dropped_cat']}] ({bw}×{bh} px)", fontsize=10)
    axes[1, 1].axis("off")
    plt.tight_layout(); plt.savefig(out_dir / "04_geometry_and_iou_audit.png", dpi=200); plt.close()

    rep_fig_dir = Path("report/figures")
    if rep_fig_dir.exists():
        for p in out_dir.glob("*.png"):
            shutil.copy2(p, rep_fig_dir / p.name)

    print(f"Đã xuất 4 biểu đồ + eda_summary.json tại: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
