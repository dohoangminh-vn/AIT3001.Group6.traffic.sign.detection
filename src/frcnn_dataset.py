import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from src.tiling import crop_tile, horizontal_tile_offsets


def get_train_transforms() -> A.Compose:
    """Augmentation cho dashcam biển báo (tuyệt đối KHÔNG dùng HorizontalFlip)."""
    return A.Compose(
        [
            A.Affine(scale=(0.9, 1.12), translate_percent=(-0.05, 0.05), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.2),
    )


class TrafficSignCocoDataset(Dataset):
    """Dataset PyTorch đọc COCO JSON cho Faster R-CNN; hỗ trợ Tiled 640x640 trên RAM và Copy-Paste lớp hiếm."""

    def __init__(
        self,
        coco_json: Path,
        img_dir: Path,
        transforms: Optional[A.Compose] = None,
        limit: Optional[int] = None,
        tiled: bool = False,
        tile_size: int = 640,
        neg_tile_ratio: float = 0.15,
        copy_paste: bool = False,
        rare_classes: Tuple[int, ...] = (3, 4, 7),
        seed: int = 42,
    ):
        coco = json.loads(Path(coco_json).read_text(encoding="utf-8"))
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        self.tiled = tiled
        self.tile_size = tile_size
        self.copy_paste = copy_paste
        self.rng = random.Random(seed)

        anns_by_img = defaultdict(list)
        for a in coco["annotations"]:
            if a["bbox"][2] > 1 and a["bbox"][3] > 1:
                anns_by_img[a["image_id"]].append(a)

        self.images = [im for im in coco["images"] if anns_by_img[im["id"]]]
        if limit:
            self.images = self.images[:limit]
        self.anns_by_img = anns_by_img

        # Lập chỉ mục các lát cắt (nếu tiled=True) — 0 MB tốn thêm ổ cứng
        self.samples: List[Tuple[int, Optional[int]]] = []
        if self.tiled:
            pos_tiles, neg_tiles = [], []
            for idx, im in enumerate(self.images):
                w_img = im.get("width", 1622)
                boxes_xyxy = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns_by_img[im["id"]]]
                for x0 in horizontal_tile_offsets(w_img, tile_size):
                    has_box = any(
                        min(b[2], x0 + tile_size) - max(b[0], x0) >= 0.6 * max(1.0, b[2] - b[0])
                        for b in boxes_xyxy
                    )
                    (pos_tiles if has_box else neg_tiles).append((idx, x0))
            self.rng.shuffle(neg_tiles)
            n_neg = min(len(neg_tiles), int(len(pos_tiles) * neg_tile_ratio / max(1e-6, 1.0 - neg_tile_ratio)))
            self.samples = pos_tiles + neg_tiles[:n_neg]
            self.rng.shuffle(self.samples)
        else:
            self.samples = [(idx, None) for idx in range(len(self.images))]

        # Kho patch lớp hiếm cho Copy-Paste (chỉ trích từ tập hiện tại, không rò rỉ val/test)
        self.patch_bank: List[Tuple[np.ndarray, int]] = []
        if self.copy_paste:
            for im in self.images:
                rare_anns = [a for a in anns_by_img[im["id"]] if a["category_id"] in rare_classes and 10 <= min(a["bbox"][2], a["bbox"][3]) <= 90]
                if not rare_anns:
                    continue
                img_np = np.array(Image.open(self.img_dir / im["file_name"]).convert("RGB"))
                for a in rare_anns:
                    x, y, w, h = [int(round(v)) for v in a["bbox"]]
                    crop = img_np[max(0, y):y + h, max(0, x):x + w]
                    if crop.shape[0] >= 8 and crop.shape[1] >= 8:
                        self.patch_bank.append((crop.copy(), int(a["category_id"])))
                    if len(self.patch_bank) >= 250:
                        break
                if len(self.patch_bank) >= 250:
                    break

    def _apply_copy_paste(self, img_np: np.ndarray, boxes: List[List[float]], labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        if not self.patch_bank or self.rng.random() > 0.45:
            return img_np, boxes, labels
        img_out = img_np.copy()
        h_img, w_img = img_out.shape[:2]
        for _ in range(self.rng.randint(1, 2)):
            patch, cat_id = self.rng.choice(self.patch_bank)
            ph, pw = patch.shape[:2]
            if pw >= w_img - 20 or ph >= h_img - 100:
                continue
            for _attempt in range(8):
                px = self.rng.randint(10, w_img - pw - 10)
                py = self.rng.randint(min(90, h_img - ph - 1), max(91, min(400, h_img - ph - 1)))
                cand = [float(px), float(py), float(px + pw), float(py + ph)]
                overlap = any(min(cand[2], b[2]) > max(cand[0], b[0]) and min(cand[3], b[3]) > max(cand[1], b[1]) for b in boxes)
                if not overlap:
                    img_out[py:py + ph, px:px + pw] = patch
                    boxes.append(cand)
                    labels.append(cat_id)
                    break
        return img_out, boxes, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_idx, x0 = self.samples[idx]
        im_info = self.images[img_idx]
        img_np = np.array(Image.open(self.img_dir / im_info["file_name"]).convert("RGB"))
        h_img, w_img = img_np.shape[:2]

        boxes, labels = [], []
        for a in self.anns_by_img[im_info["id"]]:
            x, y, w, h = a["bbox"]
            x1, y1 = max(0.0, min(float(x), w_img - 1.0)), max(0.0, min(float(y), h_img - 1.0))
            x2, y2 = max(x1 + 1.0, min(float(x + w), float(w_img))), max(y1 + 1.0, min(float(y + h), float(h_img)))
            boxes.append([x1, y1, x2, y2])
            labels.append(int(a["category_id"]))

        if x0 is not None:
            b_arr = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
            l_arr = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
            img_np, b_arr, l_arr = crop_tile(img_np, b_arr, l_arr, x0, self.tile_size)
            boxes, labels = b_arr.tolist(), l_arr.tolist()

        if self.copy_paste and len(boxes) > 0:
            img_np, boxes, labels = self._apply_copy_paste(img_np, boxes, labels)

        if self.transforms is not None:
            aug = self.transforms(image=img_np, bboxes=boxes, labels=labels)
            if len(boxes) == 0 or len(aug["bboxes"]) > 0:
                img_np = aug["image"]
                boxes = [list(b) for b in aug["bboxes"]]
                labels = [int(c) for c in aug["labels"]]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([im_info["id"]], dtype=torch.int64),
        }
        return to_tensor(img_np), target


def detection_collate_fn(batch: List):
    return tuple(zip(*batch))
