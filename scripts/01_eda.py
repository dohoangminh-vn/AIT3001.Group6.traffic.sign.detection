import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Đường dẫn đầu vào / đầu ra
DATA_DIR = Path("data/za_traffic_2020/traffic_train")
ANN_PATH = DATA_DIR / "train_traffic_sign_dataset.json"
OUT_DIR = Path("outputs/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    coco = json.loads(ANN_PATH.read_text(encoding="utf-8"))
    images = coco["images"]
    anns = coco["annotations"]
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {im["id"]: im for im in images}

    # --- 1. Thống kê tổng quan ---
    widths = np.array([a["bbox"][2] for a in anns], dtype=float)
    heights = np.array([a["bbox"][3] for a in anns], dtype=float)
    areas = widths * heights

    n_small = int((areas < 32**2).sum())
    n_med = int(((areas >= 32**2) & (areas < 96**2)).sum())
    n_large = int((areas >= 96**2).sum())

    cat_counts = Counter(a["category_id"] for a in anns)
    street_counts = Counter(im.get("street_id", -1) for im in images)

    print("=" * 60)
    print(f"Tổng số ảnh: {len(images)} | Tổng số biển báo: {len(anns)} | Số lớp: {len(cats)}")
    print(f"Trung bình biển báo / ảnh: {len(anns) / len(images):.2f}")
    print(f"Số street_id duy nhất: {len(street_counts)} (nhiều nhất: {street_counts.most_common(1)[0][1]} ảnh/đường)")
    print("-" * 60)
    print(f"Kích thước theo chuẩn COCO (area = w * h):")
    print(f"  Small  (< 32x32)       : {n_small:5d} ({n_small / len(anns) * 100:5.1f}%)")
    print(f"  Medium (32x32..96x96)  : {n_med:5d} ({n_med / len(anns) * 100:5.1f}%)")
    print(f"  Large  (>= 96x96)      : {n_large:5d} ({n_large / len(anns) * 100:5.1f}%)")
    print(f"  Trung vị (median) w x h: {np.median(widths):.1f} x {np.median(heights):.1f} px")
    print(f"  Nhỏ nhất / Lớn nhất    : {widths.min():.0f}x{heights.min():.0f} px / {widths.max():.0f}x{heights.max():.0f} px")
    print("-" * 60)
    print("Phân bố theo 7 lớp:")
    for cid in sorted(cats):
        cnt = cat_counts[cid]
        print(f"  [{cid}] {cats[cid]:<18}: {cnt:4d} ({cnt / len(anns) * 100:5.1f}%)")
    print("=" * 60)

    # Lưu tóm tắt ra JSON để sau dùng viết báo cáo ACL
    summary = {
        "num_images": len(images),
        "num_annotations": len(anns),
        "num_streets": len(street_counts),
        "coco_scale": {"small": n_small, "medium": n_med, "large": n_large},
        "median_wh": [float(np.median(widths)), float(np.median(heights))],
        "class_counts": {cats[cid]: cat_counts[cid] for cid in sorted(cats)},
    }
    (OUT_DIR / "eda_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- 2. Biểu đồ 1: Phân bố 7 lớp (thấy ngay mất cân bằng lớp 5.5x) ---
    sorted_cids = sorted(cats, key=lambda c: cat_counts[c], reverse=True)
    labels = [f"[{c}] {cats[c]}" for c in sorted_cids]
    values = [cat_counts[c] for c in sorted_cids]
    colors = ["#2b7bba" if v > 1000 else "#d95f02" for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + 35, bar.get_y() + bar.get_height() / 2,
                f"{val} ({val / len(anns) * 100:.1f}%)", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("Số lượng bounding box")
    ax.set_title("Phân bố 7 lớp biển báo (màu cam: lớp hiếm < 1,000 mẫu)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_class_distribution.png", dpi=200)
    plt.close()

    # --- 3. Biểu đồ 2: Phân bố kích thước bbox & tác động khi resize về 640 ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Trái: Scatter w vs h
    axes[0].scatter(widths, heights, s=6, alpha=0.25, color="#1f77b4")
    axes[0].axvline(32, color="red", linestyle="--", label="Ngưỡng Small (32 px)")
    axes[0].axhline(32, color="red", linestyle="--")
    axes[0].set_xlim(0, 150)
    axes[0].set_ylim(0, 150)
    axes[0].set_xlabel("Chiều rộng box gốc (px)")
    axes[0].set_ylabel("Chiều cao box gốc (px)")
    axes[0].set_title("Kích thước box gốc (zoom 0..150 px)", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Phải: Tỉ lệ Small / Medium / Large
    scale_names = ["Small\n(< 32×32)", "Medium\n(32²..96²)", "Large\n(≥ 96×96)"]
    scale_vals = [n_small, n_med, n_large]
    scale_colors = ["#e74c3c", "#f39c12", "#27ae60"]
    bars2 = axes[1].bar(scale_names, scale_vals, color=scale_colors)
    for bar, val in zip(bars2, scale_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 150,
                     f"{val}\n({val / len(anns) * 100:.1f}%)", ha="center", fontweight="bold")
    axes[1].set_ylim(0, max(scale_vals) * 1.18)
    axes[1].set_ylabel("Số lượng biển báo")
    axes[1].set_title("Tỉ lệ vật thể theo chuẩn COCO", fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_bbox_scale_analysis.png", dpi=200)
    plt.close()

    # --- 4. Biểu đồ 3: Số ảnh theo 38 street_id + Phóng to 6 biển báo các cỡ ---
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 1])

    # Hàng trên: Phân bố ảnh theo 38 street_id
    ax_top = fig.add_subplot(gs[0, :])
    top_streets = street_counts.most_common()
    s_ids = [str(k) for k, _ in top_streets]
    s_vals = [v for _, v in top_streets]
    ax_top.bar(s_ids, s_vals, color="#4c72b0")
    ax_top.set_title(f"Số khung hình trên mỗi street_id ({len(street_counts)} tuyến đường) — Lý do PHẢI chia theo nhóm!",
                     fontsize=12, fontweight="bold")
    ax_top.set_xlabel("street_id")
    ax_top.set_ylabel("Số ảnh")
    ax_top.tick_params(axis="x", rotation=60, labelsize=8)
    ax_top.grid(axis="y", linestyle="--", alpha=0.4)

    # Hàng dưới: Cắt phóng to 6 biển báo đại diện từ cực nhỏ đến lớn
    sorted_anns = sorted(anns, key=lambda a: a["bbox"][2] * a["bbox"][3])
    pick_indices = [100, 1500, 4500, 8000, 10200, 10950]
    for idx, ann_i in enumerate(pick_indices):
        a = sorted_anns[ann_i]
        im_info = img_map[a["image_id"]]
        im = Image.open(DATA_DIR / "images" / im_info["file_name"]).convert("RGB")
        x, y, w, h = [int(v) for v in a["bbox"]]
        pad = max(6, int(max(w, h) * 0.35))
        crop = im.crop((max(0, x - pad), max(0, y - pad),
                        min(im.width, x + w + pad), min(im.height, y + h + pad)))
        ax_sub = fig.add_subplot(gs[1, idx])
        ax_sub.imshow(crop)
        ax_sub.set_title(f"{cats[a['category_id']]}\n{w}×{h} px", fontsize=9)
        ax_sub.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_street_and_samples.png", dpi=200)
    plt.close()

    print(f"\nĐã xuất 3 biểu đồ + eda_summary.json vào: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()