import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


def get_train_transforms() -> A.Compose:
    """
    Augmentation cho ảnh dashcam biển báo (KHÔNG dùng HorizontalFlip để tránh đảo ngược biển rẽ/tốc độ).
    """
    return A.Compose(
        [
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_visibility=0.2
        ),
    )


class TrafficSignCocoDataset(Dataset):
    """Dataset chuẩn PyTorch đọc COCO JSON cho Faster R-CNN (nhãn 1..7, box [x1, y1, x2, y2])."""

    def __init__(
        self,
        coco_json: Path,
        img_dir: Path,
        transforms: A.Compose | None = None,
        limit: int | None = None,
    ):
        coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
        self.img_dir = Path(img_dir)
        self.transforms = transforms

        anns_by_img = defaultdict(list)
        for a in coco["annotations"]:
            if a["bbox"][2] > 1 and a["bbox"][3] > 1:
                anns_by_img[a["image_id"]].append(a)

        # Chỉ giữ ảnh có ít nhất 1 box hợp lệ
        self.images = [im for im in coco["images"] if anns_by_img[im["id"]]]
        if limit:
            self.images = self.images[:limit]
        self.anns_by_img = anns_by_img

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        im_info = self.images[idx]
        img_np = np.array(Image.open(self.img_dir / im_info["file_name"]).convert("RGB"))
        h_img, w_img = img_np.shape[:2]

        boxes, labels = [], []
        for a in self.anns_by_img[im_info["id"]]:
            x, y, w, h = a["bbox"]
            x1 = max(0.0, min(float(x), w_img - 1.0))
            y1 = max(0.0, min(float(y), h_img - 1.0))
            x2 = max(x1 + 1.0, min(float(x + w), float(w_img)))
            y2 = max(y1 + 1.0, min(float(y + h), float(h_img)))
            boxes.append([x1, y1, x2, y2])
            labels.append(int(a["category_id"]))  # Giữ 1..7 (0 là background)

        if self.transforms is not None and len(boxes) > 0:
            aug = self.transforms(image=img_np, bboxes=boxes, labels=labels)
            if len(aug["bboxes"]) > 0:
                img_np = aug["image"]
                boxes = [list(b) for b in aug["bboxes"]]
                labels = [int(c) for c in aug["labels"]]

        image_tensor = to_tensor(img_np)  # [3, H, W], float32 trong [0, 1]
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([im_info["id"]], dtype=torch.int64),
        }
        return image_tensor, target


def detection_collate_fn(batch: List):
    """Mỗi ảnh có số lượng bounding box khác nhau nên KHÔNG thể torch.stack() như bài toán phân loại."""
    return tuple(zip(*batch))