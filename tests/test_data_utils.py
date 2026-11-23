import json
from pathlib import Path

import pytest

from src.data_utils import (
    coco_bbox_to_yolo,
    coco_cat_to_yolo_cls,
    filter_coco_by_image_ids,
    split_by_street_id,
    yolo_bbox_to_coco,
    yolo_cls_to_coco_cat,
    deduplicate_annotations,
)


def test_coco_yolo_bbox_roundtrip():
    bbox = [880.0, 333.0, 19.0, 18.0]
    yolo = coco_bbox_to_yolo(bbox, img_w=1622, img_h=626)
    back = yolo_bbox_to_coco(yolo, img_w=1622, img_h=626)
    assert all(abs(a - b) < 1e-5 for a, b in zip(bbox, back))


def test_bbox_clipped_to_unit_range():
    xc, yc, wn, hn = coco_bbox_to_yolo([-5.0, 620.0, 30.0, 20.0], img_w=1622, img_h=626)
    assert 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0
    assert 0.0 <= wn <= 1.0 and 0.0 <= hn <= 1.0


def test_category_id_mapping():
    for cid in range(1, 8):
        yid = coco_cat_to_yolo_cls(cid)
        assert 0 <= yid <= 6
        assert yolo_cls_to_coco_cat(yid) == cid
    with pytest.raises(ValueError):
        coco_cat_to_yolo_cls(0)
    with pytest.raises(ValueError):
        yolo_cls_to_coco_cat(7)


def test_split_has_zero_street_leakage():
    ann_path = Path("data/za_traffic_2020/traffic_train/train_traffic_sign_dataset.json")
    if not ann_path.exists():
        pytest.skip("Chưa có dữ liệu gốc để chạy test tích hợp")
    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    splits = split_by_street_id(coco)
    img_map = {im["id"]: im for im in coco["images"]}

    # Không trùng image_id
    assert splits["train"].isdisjoint(splits["val"])
    assert splits["train"].isdisjoint(splits["test"])
    assert splits["val"].isdisjoint(splits["test"])
    assert len(splits["train"] | splits["val"] | splits["test"]) == len(coco["images"])

    # Không trùng street_id (0% Data Leakage)
    s_tr = {img_map[i]["street_id"] for i in splits["train"]}
    s_va = {img_map[i]["street_id"] for i in splits["val"]}
    s_te = {img_map[i]["street_id"] for i in splits["test"]}
    assert s_tr.isdisjoint(s_va) and s_tr.isdisjoint(s_te) and s_va.isdisjoint(s_te)

    # Cả 3 tập đều đủ 7 lớp
    for sp, ids in splits.items():
        sub = filter_coco_by_image_ids(coco, ids)
        assert len({a["category_id"] for a in sub["annotations"]}) == 7, f"Thiếu lớp ở {sp}"

    coco, n_removed = deduplicate_annotations(coco, iou_threshold=0.85)
    print(f"Đã lọc {n_removed} nhãn trùng lặp (IoU > 0.85) -> còn {len(coco['annotations'])} annotations sạch.")

def test_deduplicate_annotations():
    # Tái hiện đúng lỗi thực tế trên ảnh 10424.png (4 box đè lên nhau) + 1 box khác vị trí
    dummy = {
        "images": [{"id": 10424}],
        "categories": [{"id": 5, "name": "Cấm còn lại"}],
        "annotations": [
            {"id": 1, "image_id": 10424, "category_id": 5, "bbox": [699, 117, 26, 28]},
            {"id": 2, "image_id": 10424, "category_id": 5, "bbox": [699, 118, 26, 27]},
            {"id": 3, "image_id": 10424, "category_id": 5, "bbox": [699, 118, 26, 27]},
            {"id": 4, "image_id": 10424, "category_id": 5, "bbox": [699, 117, 26, 28]},
            {"id": 5, "image_id": 10424, "category_id": 5, "bbox": [100, 100, 30, 30]},
        ],
    }
    cleaned, removed = deduplicate_annotations(dummy, iou_threshold=0.85)
    assert removed == 3
    assert len(cleaned["annotations"]) == 2