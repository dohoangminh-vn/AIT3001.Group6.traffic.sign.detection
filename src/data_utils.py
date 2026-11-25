import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml
from sklearn.model_selection import StratifiedGroupKFold

CLASS_NAMES_ASCII = {
    0: "Cam nguoc chieu",
    1: "Cam dung va do",
    2: "Cam re",
    3: "Gioi han toc do",
    4: "Cam con lai",
    5: "Nguy hiem",
    6: "Hieu lenh",
}

CLASS_NAMES_VI = {
    1: "Cấm ngược chiều",
    2: "Cấm dừng và đỗ",
    3: "Cấm rẽ",
    4: "Giới hạn tốc độ",
    5: "Cấm còn lại",
    6: "Nguy hiểm",
    7: "Hiệu lệnh",
}


def _deep_merge(base: dict, child: dict) -> dict:
    merged = dict(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(config_path: Path, overrides: Optional[List[str]] = None) -> dict:
    """Đọc YAML có kế thừa `_base_` và hỗ trợ ghi đè dạng `key=val` hoặc `a.b=val`."""
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    base_rel = raw.pop("_base_", None)
    cfg = _deep_merge(load_config(Path(config_path).parent / base_rel), raw) if base_rel else raw
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override phải có dạng key=val, nhận được: {item}")
        key, val_str = item.split("=", 1)
        val = yaml.safe_load(val_str)
        cur = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    return cfg


def coco_cat_to_yolo_cls(category_id: int) -> int:
    if not (1 <= category_id <= 7):
        raise ValueError(f"category_id phải trong 1..7, nhận được: {category_id}")
    return category_id - 1


def yolo_cls_to_coco_cat(yolo_cls: int) -> int:
    if not (0 <= yolo_cls <= 6):
        raise ValueError(f"yolo_cls phải trong 0..6, nhận được: {yolo_cls}")
    return yolo_cls + 1


def coco_bbox_to_yolo(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    x1, y1 = max(0.0, min(float(x), float(img_w))), max(0.0, min(float(y), float(img_h)))
    x2, y2 = max(0.0, min(float(x + w), float(img_w))), max(0.0, min(float(y + h), float(img_h)))
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    return ((x1 + bw / 2.0) / img_w, (y1 + bh / 2.0) / img_h, bw / img_w, bh / img_h)


def yolo_bbox_to_coco(yolo_bbox: Tuple[float, float, float, float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    xc, yc, wn, hn = yolo_bbox
    w, h = wn * img_w, hn * img_h
    return (xc * img_w - w / 2.0, yc * img_h - h / 2.0, w, h)


def split_by_street_id(
    coco_dict: Dict, n_splits: int = 6, val_fold: int = 4, test_fold: int = 5, seed: int = 42
) -> Dict[str, Set[int]]:
    """Chia Train/Val/Test bằng StratifiedGroupKFold theo street_id (0% Data Leakage)."""
    cat_freq = Counter(a["category_id"] for a in coco_dict["annotations"])
    img_cats = defaultdict(list)
    for a in coco_dict["annotations"]:
        img_cats[a["image_id"]].append(a["category_id"])

    image_ids = [im["id"] for im in coco_dict["images"]]
    groups = [im["street_id"] for im in coco_dict["images"]]
    strat_labels = [min(img_cats[iid], key=lambda c: cat_freq[c]) if img_cats[iid] else 0 for iid in image_ids]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [val_idx for _, val_idx in sgkf.split(image_ids, strat_labels, groups)]
    val_ids = {image_ids[i] for i in folds[val_fold]}
    test_ids = {image_ids[i] for i in folds[test_fold]}
    return {"train": set(image_ids) - val_ids - test_ids, "val": val_ids, "test": test_ids}


def filter_coco_by_image_ids(coco_dict: Dict, image_ids: Set[int]) -> Dict:
    return {
        "info": coco_dict.get("info", {}),
        "categories": coco_dict["categories"],
        "images": [im for im in coco_dict["images"] if im["id"] in image_ids],
        "annotations": [a for a in coco_dict["annotations"] if a["image_id"] in image_ids],
    }


def compute_iou_xywh(box1: List[float], box2: List[float]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def audit_and_clean_coco(
    coco_dict: Dict, same_iou: float = 0.85, cross_iou: float = 0.90, min_side: float = 2.0
) -> Tuple[Dict, Dict]:
    """Lọc 2 tầng: (1) box suy biến, (2) trùng cùng lớp > same_iou, (3) xung đột khác lớp > cross_iou."""
    by_img = defaultdict(list)
    for a in coco_dict["annotations"]:
        by_img[a["image_id"]].append(a)

    cleaned, conflicts = [], []
    n_degen = n_same = n_cross = 0
    for iid, anns in by_img.items():
        kept = []
        for a in anns:
            w, h = a["bbox"][2], a["bbox"][3]
            if w < min_side or h < min_side:
                n_degen += 1
                continue
            if any(a["category_id"] == k["category_id"] and compute_iou_xywh(a["bbox"], k["bbox"]) > same_iou for k in kept):
                n_same += 1
                continue
            cross = next((k for k in kept if a["category_id"] != k["category_id"] and compute_iou_xywh(a["bbox"], k["bbox"]) > cross_iou), None)
            if cross is not None:
                n_cross += 1
                conflicts.append({"image_id": iid, "kept_cat": cross["category_id"], "dropped_cat": a["category_id"], "bbox": a["bbox"]})
                continue
            kept.append(a)
        cleaned.extend(kept)

    new_coco = dict(coco_dict)
    new_coco["annotations"] = cleaned
    report = {"removed_degenerate": n_degen, "removed_same_cls": n_same, "removed_cross_cls": n_cross, "conflicts": conflicts}
    return new_coco, report


def deduplicate_annotations(coco_dict: Dict, iou_threshold: float = 0.85) -> Tuple[Dict, int]:
    """Giữ tương thích ngược: lọc nhãn trùng cùng lớp > iou_threshold."""
    cleaned, rep = audit_and_clean_coco(coco_dict, same_iou=iou_threshold, cross_iou=1.1, min_side=0.0)
    return cleaned, rep["removed_same_cls"]


def export_split_manifest(coco_dict: Dict, splits: Dict[str, Set[int]], out_path: Path, seed: int = 42) -> Dict:
    """Xuất manifest dùng chung (split_street_id.json + train/val/test.txt) để cả nhóm dùng chung đúng 1 tập chia."""
    img_map = {im["id"]: im for im in coco_dict["images"]}
    sorted_splits = {k: sorted(v) for k, v in splits.items()}
    sha = hashlib.sha256(json.dumps(sorted_splits, sort_keys=True).encode("utf-8")).hexdigest()
    manifest = {
        "seed": seed,
        "sha256": sha,
        "counts": {k: len(v) for k, v in sorted_splits.items()},
        "street_ids": {k: sorted({img_map[i]["street_id"] for i in v}) for k, v in sorted_splits.items()},
        "splits": sorted_splits,
        "filenames": {k: [img_map[i]["file_name"] for i in v] for k, v in sorted_splits.items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for sp, names in manifest["filenames"].items():
        (out_path.parent / f"{sp}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return manifest
