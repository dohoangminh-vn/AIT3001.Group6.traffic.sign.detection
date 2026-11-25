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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import (
    CLASS_NAMES_ASCII,
    audit_and_clean_coco,
    coco_bbox_to_yolo,
    coco_cat_to_yolo_cls,
    export_split_manifest,
    filter_coco_by_image_ids,
    load_config,
    split_by_street_id,
    yolo_bbox_to_coco,
)


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst) if os.name == "nt" else os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def verify_yolo_visual(yolo_root: Path, out_img: Path) -> None:
    val_lbl_dir, val_img_dir = yolo_root / "labels" / "val", yolo_root / "images" / "val"
    chosen_lbl, best_count = None, -1
    for p in sorted(val_lbl_dir.glob("*.txt")):
        cnt = sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
        if cnt > best_count:
            best_count, chosen_lbl = cnt, p
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
        ax.add_patch(patches.Rectangle((x, y), w, h, lw=2, ec="cyan", fc="none"))
        ax.text(x, max(0, y - 5), f"[{int(cls_id)}] {CLASS_NAMES_ASCII[int(cls_id)]}", color="cyan", fontsize=9, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"))
    ax.set_title(f"Kiểm chứng đọc ngược nhãn YOLO: {img_path.name} ({best_count} biển báo)", fontweight="bold")
    ax.axis("off")
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_img, dpi=150); plt.close()


def main():
    parser = argparse.ArgumentParser(description="Làm sạch nhãn & chia dữ liệu theo street_id -> COCO, YOLO & Manifest chung")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("opts", nargs="*", help="Ghi đè cấu hình dạng key=val")
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["data"]["processed_dir"])
    raw_img_dir = raw_dir / "images"

    coco_raw = json.loads((raw_dir / "train_traffic_sign_dataset.json").read_text(encoding="utf-8"))
    coco, audit = audit_and_clean_coco(
        coco_raw, cfg["data"]["dedup_iou"], cfg["data"]["conflict_iou"], cfg["data"]["min_box_side"]
    )
    print(f"Làm sạch 2 tầng: trùng cùng lớp={audit['removed_same_cls']}, xung đột khác lớp={audit['removed_cross_cls']}, "
          f"suy biến={audit['removed_degenerate']} -> còn {len(coco['annotations'])} annotations sạch.")

    manifest_path = Path(cfg["data"]["split_manifest"])
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        splits = {k: set(v) for k, v in saved["splits"].items()}
        print(f"Dùng lại split cố định từ {manifest_path} (SHA256={saved.get('sha256', '')[:12]})")
    else:
        splits = split_by_street_id(
            coco, cfg["data"]["n_splits"], cfg["data"]["val_fold"], cfg["data"]["test_fold"], cfg["seed"]
        )
    img_map = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    streets = {sp: {img_map[i]["street_id"] for i in ids} for sp, ids in splits.items()}
    assert streets["train"].isdisjoint(streets["val"]) and streets["train"].isdisjoint(streets["test"]) and streets["val"].isdisjoint(streets["test"])

    manifest = export_split_manifest(coco, splits, out_dir / "split_street_id.json", seed=cfg["seed"])
    coco_out, yolo_out = Path(cfg["data"]["coco_dir"]), Path(cfg["data"]["yolo_dir"])
    coco_out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"{'Tập':<8} | {'Số ảnh':>7} | {'Tỉ lệ':>6} | {'Streets':>7} | {'Số box':>7} | {'Lớp 3 (hiếm)':>12}")
    print("-" * 72)
    total_imgs = len(coco["images"])
    for sp in ("train", "val", "test"):
        ids = splits[sp]
        sub_coco = filter_coco_by_image_ids(coco, ids)
        (coco_out / f"{sp}.json").write_text(json.dumps(sub_coco, ensure_ascii=False), encoding="utf-8")

        sp_img_dir, sp_lbl_dir = yolo_out / "images" / sp, yolo_out / "labels" / sp
        sp_img_dir.mkdir(parents=True, exist_ok=True); sp_lbl_dir.mkdir(parents=True, exist_ok=True)
        for iid in ids:
            im = img_map[iid]
            link_or_copy(raw_img_dir / im["file_name"], sp_img_dir / im["file_name"])
            lines = []
            for a in anns_by_img[iid]:
                xc, yc, wn, hn = coco_bbox_to_yolo(a["bbox"], im["width"], im["height"])
                if wn > 0 and hn > 0:
                    lines.append(f"{coco_cat_to_yolo_cls(a['category_id'])} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            (sp_lbl_dir / f"{Path(im['file_name']).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        cc = Counter(a["category_id"] for a in sub_coco["annotations"])
        print(f"{sp:<8} | {len(ids):7d} | {len(ids)/total_imgs*100:5.1f}% | {len(streets[sp]):7d} | {len(sub_coco['annotations']):7d} | {cc[3]:12d}")
    print("=" * 72)

    yaml_path = yolo_out / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump({"path": str(yolo_out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": CLASS_NAMES_ASCII}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    verify_yolo_visual(yolo_out, Path("outputs/ch2_verify_yolo_labels.jpg"))
    print(f"Manifest chung SHA256={manifest['sha256'][:12]} -> {(out_dir / 'split_street_id.json').resolve()}")


if __name__ == "__main__":
    main()
