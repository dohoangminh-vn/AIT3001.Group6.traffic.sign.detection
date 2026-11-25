import math
from typing import List, Tuple

import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou


def horizontal_tile_offsets(img_w: int = 1622, tile_w: int = 640, min_overlap: int = 128) -> List[int]:
    """Sinh tọa độ x0 cho các lát cắt 1 hàng ngang (1622px -> [0, 491, 982], chồng lấp 149px)."""
    if img_w <= tile_w:
        return [0]
    step_max = max(1, tile_w - min_overlap)
    n_tiles = max(2, math.ceil((img_w - min_overlap) / step_max))
    return [round(i * (img_w - tile_w) / (n_tiles - 1)) for i in range(n_tiles)]


def crop_tile(
    img_np: np.ndarray,
    boxes_xyxy: np.ndarray,
    labels: np.ndarray,
    x0: int,
    tile_size: int = 640,
    min_vis: float = 0.60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cắt 1 lát [tile_size x tile_size], pad nền 114 nếu thiếu, lọc box bị cắt mép < min_vis."""
    h, w = img_np.shape[:2]
    x1_end = min(w, x0 + tile_size)
    y1_end = min(h, tile_size)
    patch = img_np[0:y1_end, x0:x1_end]
    ph, pw = patch.shape[:2]
    if ph < tile_size or pw < tile_size:
        canvas = np.full((tile_size, tile_size, 3), 114, dtype=img_np.dtype)
        canvas[:ph, :pw] = patch
        patch = canvas

    if len(boxes_xyxy) == 0:
        return patch, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    orig_area = np.maximum(1.0, (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]))
    cx1 = np.clip(boxes_xyxy[:, 0], x0, x1_end) - x0
    cy1 = np.clip(boxes_xyxy[:, 1], 0, y1_end)
    cx2 = np.clip(boxes_xyxy[:, 2], x0, x1_end) - x0
    cy2 = np.clip(boxes_xyxy[:, 3], 0, y1_end)
    bw, bh = np.maximum(0.0, cx2 - cx1), np.maximum(0.0, cy2 - cy1)
    vis = (bw * bh) / orig_area
    keep = (vis >= min_vis) & (bw >= 2.0) & (bh >= 2.0)
    kept_boxes = np.stack([cx1[keep], cy1[keep], cx2[keep], cy2[keep]], axis=1).astype(np.float32)
    return patch, kept_boxes, labels[keep].astype(np.int64)


def merge_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    method: str = "wbf",
    iou_thr: float = 0.55,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hợp nhất dự đoán từ nhiều lát cắt + ảnh gốc bằng Batched NMS hoặc Weighted Boxes Fusion."""
    if boxes.numel() == 0:
        return boxes, scores, labels

    if len(scores) > 400:
        top = torch.topk(scores, 400).indices
        boxes, scores, labels = boxes[top], scores[top], labels[top]

    if method == "nms":
        keep = batched_nms(boxes, scores, labels, iou_thr)
        return boxes[keep], scores[keep], labels[keep]

    if method != "wbf":
        raise ValueError(f"Unsupported merge method: {method}")

    out_boxes, out_scores, out_labels = [], [], []
    for cls_id in labels.unique():
        mask = labels == cls_id
        c_boxes, c_scores = boxes[mask], scores[mask]
        order = torch.argsort(c_scores, descending=True)
        c_boxes, c_scores = c_boxes[order], c_scores[order]

        clusters_boxes: List[torch.Tensor] = []
        clusters_weights: List[float] = []
        clusters_max_s: List[float] = []

        for b, s in zip(c_boxes, c_scores):
            s_val = float(s.item())
            if not clusters_boxes:
                clusters_boxes.append(b.clone())
                clusters_weights.append(s_val)
                clusters_max_s.append(s_val)
                continue
            stack_c = torch.stack(clusters_boxes)
            ious = box_iou(b.unsqueeze(0), stack_c).squeeze(0)
            best_idx = int(torch.argmax(ious).item())
            if float(ious[best_idx].item()) >= iou_thr:
                w_old = clusters_weights[best_idx]
                w_new = w_old + s_val
                clusters_boxes[best_idx] = (clusters_boxes[best_idx] * w_old + b * s_val) / max(w_new, 1e-6)
                clusters_weights[best_idx] = w_new
                clusters_max_s[best_idx] = max(clusters_max_s[best_idx], s_val)
            else:
                clusters_boxes.append(b.clone())
                clusters_weights.append(s_val)
                clusters_max_s.append(s_val)

        out_boxes.append(torch.stack(clusters_boxes))
        out_scores.append(torch.tensor(clusters_max_s, dtype=scores.dtype, device=boxes.device))
        out_labels.append(torch.full((len(clusters_boxes),), cls_id, dtype=labels.dtype, device=boxes.device))

    final_boxes = torch.cat(out_boxes, dim=0)
    final_scores = torch.cat(out_scores, dim=0)
    final_labels = torch.cat(out_labels, dim=0)
    order = torch.argsort(final_scores, descending=True)
    return final_boxes[order], final_scores[order], final_labels[order]

