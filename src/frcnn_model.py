from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


def build_faster_rcnn(
    num_classes: int = 8,
    small_anchors: bool = True,
    min_size: int = 626,
    max_size: int = 1622,
    freeze_backbone: bool = False,
):
    """
    Khởi tạo Faster R-CNN ResNet50-FPN-v2 pretrained COCO và tùy chỉnh cho biển báo nhỏ Việt Nam:
    - Thay đầu phân loại (Box Predictor) sang 8 lớp (0: background + 7 lớp biển báo).
    - Tùy chọn small_anchors=True: hạ kích thước anchor từ (32..512) xuống (8..128)
      (giữ nguyên 3 aspect ratios mỗi tầng FPN nên tương thích 100% với trọng số RPN pretrained!).
    """
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

    # 1. Thay Box Predictor sang num_classes=8
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 2. Chỉnh kích thước đầu vào nội bộ (giữ nguyên độ phân giải rộng 1622x626 trên GPU)
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size

    # 3. Tùy chỉnh Anchor nhỏ cho 5 tầng FPN (P2..P6) — điểm nhấn kỹ thuật cho 78.7% biển báo nhỏ!
    if small_anchors:
        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        model.rpn.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes, aspect_ratios=aspect_ratios
        )

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    return model