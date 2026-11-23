import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import yaml
from PIL import Image

# Cho phép import src.* khi chạy trực tiếp từ thư mục gốc btl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import (
    CLASS_NAMES_ASCII,
    coco_bbox_to_yolo,
    coco_cat_to_yolo_cls,
    filter_coco_by_image_ids,
    split_by_street_id,
    yolo_bbox_to_coco,
    deduplicate_annotations,
)


def link_or_copy(src: Path, dst: Path) -> None:
    """Tạo liên kết không tốn dung lượng ổ cứng (hardlink trên Windows, symlink trên Linux/Kaggle)."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        if os.name == "nt":
            os.link(src, dst)       # Windows NTFS cùng ổ đĩa: 0 MB tốn thêm, không cần quyền Admin
        else:
            os.symlink(src.resolve(), dst)  # Linux / Kaggle: trỏ thẳng vào /kaggle/input
    except OSError:
        shutil.copy2(src, dst)


def verify_yolo_visual(yolo_root: Path, out_img: Path) -> None:
    """Đọc NGƯỢC file .txt YOLO vừa sinh ra và vẽ lên ảnh để mắt thường kiểm chứng 100% khớp."""
    val_lbl_dir = yolo_root / "labels" / "val"
    val_img_dir = yolo_root / "images" / "val"

    # Chọn 1 ảnh trong tập val có nhiều biển báo (>= 3) cho dễ nhìn
    chosen_lbl = None
    best_count = -1
    for p in sorted(val_lbl_dir.glob("*.txt")):
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) > best_count:
            best_count = len(lines)
            chosen_lbl = p
        if best_count >= 4:
            break

    img_path = val_img_dir / f"{chosen_lbl.stem}.png"
    img = Image.open(img_path).convert("RGB")
    w_img, h_img = img.size

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(img)
    for line in chosen_lbl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls_id, xc, yc, wn, hn = map(float, line.split())
        x, y, w, h = yolo_bbox_to_coco((xc, yc, wn, hn), w_img, h_img)
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="cyan", facecolor="none")
        ax.add_patch(rect)
        ax.text(
            x, max(0, y - 5), f"[{int(cls_id)}] {CLASS_NAMES_ASCII[int(cls_id)]}",
            color="cyan", fontsize=9, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"),
        )
    ax.set_title(f"Kiểm chứng đọc ngược nhãn YOLO: {img_path.name} ({best_count} biển báo)", fontweight="bold")
    ax.axis("off")
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_img, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Chia dữ liệu theo street_id -> YOLO & COCO")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/za_traffic_2020/traffic_train"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    ann_path = args.raw_dir / "train_traffic_sign_dataset.json"
    raw_img_dir = args.raw_dir / "images"
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    coco, n_removed = deduplicate_annotations(coco, iou_threshold=0.85)
    print(f"Đã lọc {n_removed} nhãn trùng lặp (IoU > 0.85) -> còn {len(coco['annotations'])} annotations sạch.")

    # 1. Chia dữ liệu chống Data Leakage theo street_id
    splits = split_by_street_id(coco, n_splits=6, val_fold=4, test_fold=5, seed=42)
    img_map = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # Kiểm tra cứng: tuyệt đối KHÔNG có street_id nào trùng giữa train/val/test
    streets = {sp: {img_map[i]["street_id"] for i in ids} for sp, ids in splits.items()}
    assert streets["train"].isdisjoint(streets["val"]), "LỖI: Trùng street_id giữa train và val!"
    assert streets["train"].isdisjoint(streets["test"]), "LỖI: Trùng street_id giữa train và test!"
    assert streets["val"].isdisjoint(streets["test"]), "LỖI: Trùng street_id giữa val và test!"

    coco_out = args.out_dir / "coco"
    yolo_out = args.out_dir / "yolo"
    coco_out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"{'Tập':<8} | {'Số ảnh':>7} | {'Tỉ lệ':>6} | {'Streets':>7} | {'Số box':>7} | {'Lớp 3 (hiếm)':>12}")
    print("-" * 72)

    total_imgs = len(coco["images"])
    for sp in ("train", "val", "test"):
        ids = splits[sp]
        # 2. Xuất COCO JSON (dùng cho Faster R-CNN và pycocotools eval)
        sub_coco = filter_coco_by_image_ids(coco, ids)
        (coco_out / f"{sp}.json").write_text(
            json.dumps(sub_coco, ensure_ascii=False), encoding="utf-8"
        )

        # 3. Xuất YOLO images (hardlink/symlink) + labels (.txt)
        sp_img_dir = yolo_out / "images" / sp
        sp_lbl_dir = yolo_out / "labels" / sp
        sp_img_dir.mkdir(parents=True, exist_ok=True)
        sp_lbl_dir.mkdir(parents=True, exist_ok=True)

        for iid in ids:
            im = img_map[iid]
            src_img = raw_img_dir / im["file_name"]
            dst_img = sp_img_dir / im["file_name"]
            link_or_copy(src_img, dst_img)

            lines = []
            for a in anns_by_img[iid]:
                cls_id = coco_cat_to_yolo_cls(a["category_id"])
                xc, yc, wn, hn = coco_bbox_to_yolo(a["bbox"], im["width"], im["height"])
                if wn > 0 and hn > 0:
                    lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            lbl_path = sp_lbl_dir / f"{Path(im['file_name']).stem}.txt"
            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        cc = Counter(a["category_id"] for a in sub_coco["annotations"])
        print(
            f"{sp:<8} | {len(ids):7d} | {len(ids)/total_imgs*100:5.1f}% | "
            f"{len(streets[sp]):7d} | {len(sub_coco['annotations']):7d} | {cc[3]:12d}"
        )

    print("=" * 72)
    print("Kiểm chứng Data Leakage -> Overlap street_id giữa Train/Val/Test = 0 (AN TOÀN!)")

    # 4. Tạo file dataset.yaml có đường dẫn tuyệt đối cho máy hiện tại (local hoặc Kaggle)
    dataset_yaml = {
        "path": str(yolo_out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES_ASCII,
    }
    yaml_path = yolo_out / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # 5. Vẽ 1 ảnh kiểm chứng đọc ngược nhãn YOLO
    verify_img = Path("outputs/ch2_verify_yolo_labels.jpg")
    verify_yolo_visual(yolo_out, verify_img)

    print(f"Đã tạo COCO JSON tại   : {coco_out.resolve()}")
    print(f"Đã tạo YOLO YAML tại   : {yaml_path.resolve()}")
    print(f"Đã lưu ảnh kiểm chứng  : {verify_img.resolve()}")


if __name__ == "__main__":
    main()