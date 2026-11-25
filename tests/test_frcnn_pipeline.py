from pathlib import Path

import numpy as np
import torch

from src.coco_eval import analyze_detection_errors
from src.data_utils import audit_and_clean_coco, export_split_manifest, load_config
from src.frcnn_model import build_faster_rcnn, compute_cb_weights, giou_loss_boxes
from src.tiling import crop_tile, horizontal_tile_offsets, merge_detections


def test_load_config_inheritance_and_overrides():
    cfg = load_config(Path("configs/frcnn_baseline.yaml"), ["anchor_mode=default", "lr=0.0005", "data.dedup_iou=0.9"])
    assert cfg["seed"] == 42
    assert cfg["anchor_mode"] == "default"
    assert cfg["lr"] == 0.0005
    assert cfg["data"]["dedup_iou"] == 0.9


def test_audit_cross_class_and_export_manifest(tmp_path):
    dummy = {
        "images": [
            {"id": 1, "file_name": "1.png", "street_id": 10},
            {"id": 2, "file_name": "2.png", "street_id": 20},
        ],
        "categories": [{"id": i, "name": str(i)} for i in range(1, 8)],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [100, 100, 20, 20]},
            {"id": 2, "image_id": 1, "category_id": 5, "bbox": [100, 100, 20, 20]},  # xung đột khác lớp
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [10, 10, 1, 1]},      # suy biến < 2px
        ],
    }
    cleaned, rep = audit_and_clean_coco(dummy, same_iou=0.85, cross_iou=0.90, min_side=2.0)
    assert rep["removed_cross_cls"] == 1
    assert rep["removed_degenerate"] == 1
    assert len(cleaned["annotations"]) == 1

    manifest = export_split_manifest(dummy, {"train": {1}, "val": {2}, "test": set()}, tmp_path / "split.json")
    assert manifest["counts"] == {"train": 1, "val": 1, "test": 0}
    assert (tmp_path / "train.txt").read_text(encoding="utf-8").strip() == "1.png"


def test_tiling_offsets_crop_and_wbf_merge():
    offsets = horizontal_tile_offsets(1622, 640)
    assert offsets == [0, 491, 982]

    img = np.zeros((626, 1622, 3), dtype=np.uint8)
    boxes = np.array([[520.0, 200.0, 560.0, 240.0], [635.0, 200.0, 665.0, 240.0]], dtype=np.float32)
    labels = np.array([3, 4], dtype=np.int64)
    patch, tb, tl = crop_tile(img, boxes, labels, x0=0, tile_size=640, min_vis=0.60)
    assert patch.shape == (640, 640, 3)
    assert tl.tolist() == [3]  # Box thứ 2 bị cắt mép 640 chỉ còn 5/30 = 16.7% -> loại bỏ
    assert tb[0].tolist() == [520.0, 200.0, 560.0, 240.0]

    # Hợp nhất 2 hộp trùng nhau ở vùng chồng lấp bằng WBF
    m_b, m_s, m_l = merge_detections(
        torch.tensor([[520.0, 200.0, 560.0, 240.0], [522.0, 201.0, 562.0, 241.0]]),
        torch.tensor([0.9, 0.7]),
        torch.tensor([3, 3]),
        method="wbf",
    )
    assert len(m_b) == 1 and int(m_l[0]) == 3 and abs(float(m_s[0]) - 0.9) < 1e-5


def test_custom_frcnn_cb_focal_giou_and_roi14():
    weights = compute_cb_weights({1: 1000, 2: 2000, 3: 300, 4: 600, 5: 1200, 6: 2500, 7: 700})
    assert weights[3] > weights[6]  # Lớp hiếm Cấm rẽ > lớp phổ biến Nguy hiểm

    b1 = torch.tensor([[10.0, 10.0, 30.0, 30.0]], requires_grad=True)
    b2 = torch.tensor([[12.0, 12.0, 32.0, 32.0]])
    g_loss = giou_loss_boxes(b1, b2)
    assert 0.0 < float(g_loss.item()) < 1.0
    g_loss.backward()
    assert b1.grad is not None and torch.isfinite(b1.grad).all()

    model = build_faster_rcnn(
        num_classes=8, anchor_mode="small_sign_ar", roi_size=14, loss_type="cb_focal",
        use_giou=True, min_size=128, max_size=256, pretrained=False,
    )
    model.train()
    imgs = [torch.rand(3, 128, 256), torch.rand(3, 128, 256)]
    tgts = [
        {"boxes": torch.tensor([[40.0, 40.0, 60.0, 60.0]]), "labels": torch.tensor([3])},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},  # Hard-negative tile
    ]
    losses = model(imgs, tgts)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert model.roi_heads.box_predictor.cls_score.weight.grad is not None

    model.eval()
    with torch.no_grad():
        preds = model(imgs[:1])
    assert len(preds) == 1 and "boxes" in preds[0]

    # Giả lập float16 AMP trên tọa độ ảnh thật 1622x626 (tránh tràn area > 65504 trong BoxCoder/GIoU)
    logits_fp16 = torch.randn(16, 8, dtype=torch.float16, requires_grad=True)
    reg_fp16 = torch.randn(16, 32, dtype=torch.float16, requires_grad=True)
    props_full = [torch.tensor([[100.0, 100.0, 1500.0, 550.0]] * 16, dtype=torch.float32)]
    lbls_full = [torch.randint(0, 8, (16,), dtype=torch.int64)]
    reg_t_full = [torch.randn(16, 4, dtype=torch.float32)]
    lc, lb = model.roi_heads._compute_custom_loss(logits_fp16, reg_fp16, props_full, lbls_full, reg_t_full)
    assert torch.isfinite(lc) and torch.isfinite(lb)
    (lc + lb).backward()
    assert torch.isfinite(logits_fp16.grad).all() and torch.isfinite(reg_fp16.grad).all()


def test_analyze_detection_errors_4way():
    gt = {
        "images": [{"id": 1}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},  # TP
            {"id": 2, "image_id": 1, "category_id": 3, "bbox": [60, 60, 20, 20]},  # Miscls -> predicted as 5
            {"id": 3, "image_id": 1, "category_id": 4, "bbox": [120, 120, 15, 15]},  # FN_small
        ],
    }
    preds = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 1, "category_id": 5, "bbox": [60, 60, 20, 20], "score": 0.8},
        {"image_id": 1, "category_id": 2, "bbox": [200, 200, 25, 25], "score": 0.7},  # FP_background
    ]
    err = analyze_detection_errors(gt, preds, conf_thresh=0.25)
    assert (err["TP"], err["Misclassification"], err["FP_background"], err["FN_small"]) == (1, 1, 1, 1)

