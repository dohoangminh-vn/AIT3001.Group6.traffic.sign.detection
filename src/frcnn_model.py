from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead, FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


def compute_cb_weights(
    class_counts: Optional[Dict[int, int]] = None,
    num_classes: int = 8,
    beta: float = 0.999,
    bg_weight: float = 0.35,
) -> torch.Tensor:
    """Trọng số cân bằng lớp theo Effective Number of Samples (Cui et al., CVPR 2019)."""
    if not class_counts:
        return torch.tensor([bg_weight] + [1.0] * (num_classes - 1), dtype=torch.float32)
    fg = torch.tensor(
        [(1.0 - beta) / (1.0 - beta ** max(1, int(class_counts.get(c, 1)))) for c in range(1, num_classes)],
        dtype=torch.float32,
    )
    fg = fg / fg.mean().clamp(min=1e-6)
    return torch.cat([torch.tensor([bg_weight], dtype=torch.float32), fg])


def giou_loss_boxes(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Generalized IoU loss theo từng cặp box tương ứng [N, 4] (tự cài đặt)."""
    ix1, iy1 = torch.maximum(pred_boxes[:, 0], gt_boxes[:, 0]), torch.maximum(pred_boxes[:, 1], gt_boxes[:, 1])
    ix2, iy2 = torch.minimum(pred_boxes[:, 2], gt_boxes[:, 2]), torch.minimum(pred_boxes[:, 3], gt_boxes[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    area_g = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
    union = (area_p + area_g - inter).clamp(min=1e-7)
    iou = inter / union

    cx1, cy1 = torch.minimum(pred_boxes[:, 0], gt_boxes[:, 0]), torch.minimum(pred_boxes[:, 1], gt_boxes[:, 1])
    cx2, cy2 = torch.maximum(pred_boxes[:, 2], gt_boxes[:, 2]), torch.maximum(pred_boxes[:, 3], gt_boxes[:, 3])
    c_area = ((cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)).clamp(min=1e-7)
    return 1.0 - (iou - (c_area - union) / c_area)


class FocalRoIHeads(RoIHeads):
    """Kế thừa RoIHeads: thay Cross-Entropy mặc định bằng Class-Balanced Focal Loss + GIoU (tự cài)."""

    def __init__(self, base: RoIHeads, class_weights: torch.Tensor, gamma: float = 2.0, use_giou: bool = True):
        super().__init__(
            box_roi_pool=base.box_roi_pool,
            box_head=base.box_head,
            box_predictor=base.box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=base.box_coder.weights,
            score_thresh=base.score_thresh,
            nms_thresh=base.nms_thresh,
            detections_per_img=base.detections_per_img,
        )
        self.proposal_matcher = base.proposal_matcher
        self.fg_bg_sampler = base.fg_bg_sampler
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma
        self.use_giou = use_giou

    def _compute_custom_loss(self, class_logits, box_regression, proposals, labels, regression_targets):
        # Ép FP32 cho bước tính loss để tránh tràn số float16 trong BoxCoder.decode_single & GIoU khi bật AMP
        class_logits = class_logits.float()
        box_regression = box_regression.float()
        labels_cat = torch.cat(labels, dim=0)
        reg_cat = torch.cat(regression_targets, dim=0).float()
        props_cat = torch.cat(proposals, dim=0).float()

        ce = F.cross_entropy(class_logits, labels_cat, reduction="none", label_smoothing=0.02)
        pt = torch.softmax(class_logits, dim=-1).gather(1, labels_cat.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0 - 1e-6)
        alpha = self.class_weights[labels_cat]
        pos_inds = torch.where(labels_cat > 0)[0]
        # Chuẩn hóa theo (num_pos + 16) để giữ biên độ loss cân bằng khi phần lớn background có (1-pt)^gamma ~ 0
        loss_cls = (alpha * ((1.0 - pt) ** self.gamma) * ce).sum() / max(1.0, float(pos_inds.numel() + 16))

        n_roi = class_logits.shape[0]
        box_reg = box_regression.reshape(n_roi, box_regression.size(-1) // 4, 4)
        if pos_inds.numel() == 0:
            loss_box = box_reg.sum() * 0.0
        else:
            pos_lbls = labels_cat[pos_inds]
            smooth_l1 = F.smooth_l1_loss(box_reg[pos_inds, pos_lbls], reg_cat[pos_inds], beta=1 / 9, reduction="sum") / labels_cat.numel()
            if self.use_giou:
                pred_b = self.box_coder.decode_single(box_reg[pos_inds, pos_lbls], props_cat[pos_inds])
                gt_b = self.box_coder.decode_single(reg_cat[pos_inds], props_cat[pos_inds])
                loss_box = smooth_l1 + 0.5 * giou_loss_boxes(pred_b, gt_b).mean()
            else:
                loss_box = smooth_l1
        return loss_cls, loss_box

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, _, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = regression_targets = None

        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)

        result, losses = [], {}
        if self.training:
            loss_cls, loss_box = self._compute_custom_loss(class_logits, box_regression, proposals, labels, regression_targets)
            losses = {"loss_classifier": loss_cls, "loss_box_reg": loss_box}
        else:
            boxes, scores, pred_labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            result = [{"boxes": b, "labels": l, "scores": s} for b, l, s in zip(boxes, pred_labels, scores)]
        return result, losses


def build_faster_rcnn(
    num_classes: int = 8,
    backbone: str = "resnet50",
    anchor_mode: str = "small_sign_ar",
    small_anchors: Optional[bool] = None,
    roi_size: int = 14,
    loss_type: str = "cb_focal",
    focal_gamma: float = 2.0,
    use_giou: bool = True,
    class_counts: Optional[Dict[int, int]] = None,
    min_size: int = 626,
    max_size: int = 1622,
    freeze_backbone: bool = False,
    pretrained: bool = True,
):
    """Khởi tạo Faster R-CNN tùy biến cho biển báo nhỏ Việt Nam."""
    if small_anchors is not None:
        anchor_mode = "small" if small_anchors else "default"

    if backbone == "mobilenet_v3":
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=None)
    else:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size

    n_levels = len(model.rpn.anchor_generator.sizes)
    if anchor_mode != "default":
        sizes = ((8,), (16,), (32,), (64,), (128,))[:n_levels] if n_levels >= 5 else ((16, 32, 64, 128),) * n_levels
        ar = (0.75, 1.0, 1.30) if anchor_mode == "small_sign_ar" else (0.5, 1.0, 2.0)
        model.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=(ar,) * len(sizes))

    # Tối ưu RPN cho ảnh đông biển nhỏ & nâng RoIAlign 14x14 bảo toàn trọng số pretrained FC
    model.rpn._pre_nms_top_n = {"training": 3000, "testing": 1500}
    model.rpn._post_nms_top_n = {"training": 1500, "testing": 1000}
    model.rpn.fg_iou_thresh = 0.6
    model.rpn.proposal_matcher.high_threshold = 0.6
    model.roi_heads.detections_per_img = 200

    if roi_size > 7 and isinstance(model.roi_heads.box_head, FastRCNNConvFCHead):
        model.roi_heads.fg_bg_sampler.batch_size_per_image = 256
        model.roi_heads.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=model.roi_heads.box_roi_pool.featmap_names, output_size=roi_size, sampling_ratio=2
        )
        layers = list(model.roi_heads.box_head.children())
        flat_idx = next(i for i, m in enumerate(layers) if isinstance(m, nn.Flatten))
        model.roi_heads.box_head = nn.Sequential(*layers[:flat_idx], nn.AdaptiveAvgPool2d((7, 7)), *layers[flat_idx:])

    if loss_type == "cb_focal":
        cb_w = compute_cb_weights(class_counts, num_classes=num_classes)
        model.roi_heads = FocalRoIHeads(model.roi_heads, cb_w, gamma=focal_gamma, use_giou=use_giou)

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    return model
