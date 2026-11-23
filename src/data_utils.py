from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

from sklearn.model_selection import StratifiedGroupKFold

# Tên không dấu dùng cho YOLO/OpenCV (tránh lỗi font ???), tên có dấu dùng cho báo cáo/Matplotlib
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


def coco_cat_to_yolo_cls(category_id: int) -> int:
    """Ánh xạ ID lớp COCO (1..7) -> YOLO (0..6)."""
    if not (1 <= category_id <= 7):
        raise ValueError(f"category_id phải trong 1..7, nhận được: {category_id}")
    return category_id - 1


def yolo_cls_to_coco_cat(yolo_cls: int) -> int:
    """Ánh xạ ngược ID lớp YOLO (0..6) -> COCO (1..7)."""
    if not (0 <= yolo_cls <= 6):
        raise ValueError(f"yolo_cls phải trong 0..6, nhận được: {yolo_cls}")
    return yolo_cls + 1


def coco_bbox_to_yolo(
    bbox: List[float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Đổi [x_min, y_min, w, h] (pixel) -> [x_center, y_center, w, h] chuẩn hóa [0, 1]."""
    x, y, w, h = bbox
    x1 = max(0.0, min(float(x), float(img_w)))
    y1 = max(0.0, min(float(y), float(img_h)))
    x2 = max(0.0, min(float(x + w), float(img_w)))
    y2 = max(0.0, min(float(y + h), float(img_h)))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = (x1 + bw / 2.0) / img_w
    yc = (y1 + bh / 2.0) / img_h
    return (xc, yc, bw / img_w, bh / img_h)


def yolo_bbox_to_coco(
    yolo_bbox: Tuple[float, float, float, float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """Đổi ngược YOLO [xc, yc, wn, hn] -> COCO [x_min, y_min, w, h] (pixel)."""
    xc, yc, wn, hn = yolo_bbox
    w = wn * img_w
    h = hn * img_h
    x = xc * img_w - w / 2.0
    y = yc * img_h - h / 2.0
    return (x, y, w, h)


def split_by_street_id(
    coco_dict: Dict,
    n_splits: int = 6,
    val_fold: int = 4,
    test_fold: int = 5,
    seed: int = 42,
) -> Dict[str, Set[int]]:
    """
    Chia Train/Val/Test (~70/16/14%) bằng StratifiedGroupKFold theo street_id:
    - Mọi khung hình cùng 1 street_id nằm trọn trong 1 tập (0% Data Leakage).
    - Gán nhãn đại diện của mỗi ảnh là lớp HIẾM NHẤT trong ảnh để cân bằng cả lớp hiếm (Cấm rẽ).
    """
    cat_freq = Counter(a["category_id"] for a in coco_dict["annotations"])
    img_cats = defaultdict(list)
    for a in coco_dict["annotations"]:
        img_cats[a["image_id"]].append(a["category_id"])

    image_ids = [im["id"] for im in coco_dict["images"]]
    groups = [im["street_id"] for im in coco_dict["images"]]
    strat_labels = [
        min(img_cats[iid], key=lambda c: cat_freq[c]) if img_cats[iid] else 0
        for iid in image_ids
    ]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [val_idx for _, val_idx in sgkf.split(image_ids, strat_labels, groups)]

    val_ids = {image_ids[i] for i in folds[val_fold]}
    test_ids = {image_ids[i] for i in folds[test_fold]}
    train_ids = set(image_ids) - val_ids - test_ids

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def filter_coco_by_image_ids(coco_dict: Dict, image_ids: Set[int]) -> Dict:
    """Trích xuất tập con COCO JSON chỉ chứa các image_id đã chọn."""
    return {
        "info": coco_dict.get("info", {}),
        "categories": coco_dict["categories"],
        "images": [im for im in coco_dict["images"] if im["id"] in image_ids],
        "annotations": [a for a in coco_dict["annotations"] if a["image_id"] in image_ids],
    }

def compute_iou_xywh(box1: List[float], box2: List[float]) -> float:
    """Tính IoU giữa 2 box định dạng COCO [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def deduplicate_annotations(coco_dict: Dict, iou_threshold: float = 0.85) -> Tuple[Dict, int]:
    """Loại bỏ các nhãn trùng/gần trùng (cùng ảnh, cùng lớp, IoU > iou_threshold)."""
    by_img = defaultdict(list)
    for a in coco_dict["annotations"]:
        by_img[a["image_id"]].append(a)

    cleaned = []
    removed = 0
    for anns in by_img.values():
        kept = []
        for a in anns:
            is_dup = any(
                a["category_id"] == k["category_id"]
                and compute_iou_xywh(a["bbox"], k["bbox"]) > iou_threshold
                for k in kept
            )
            if is_dup:
                removed += 1
            else:
                kept.append(a)
        cleaned.extend(kept)

    new_coco = dict(coco_dict)
    new_coco["annotations"] = cleaned
    return new_coco, removed