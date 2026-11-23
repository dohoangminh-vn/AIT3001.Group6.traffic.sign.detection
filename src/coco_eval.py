import contextlib
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.data_utils import CLASS_NAMES_VI


def evaluate_coco_predictions(
    gt_json: Path,
    pred_json: Path,
    img_ids: Optional[List[int]] = None,
) -> Dict:
    """
    Chấm điểm file dự đoán COCO JSON bằng pycocotools chuẩn.
    Trả về dict chứa: mAP@0.5:0.95, mAP@0.5, mAP@0.75, AP_small, AP_medium, AP_large và AP từng lớp.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = json.loads(Path(gt_json).read_text(encoding="utf-8"))
        coco_gt.createIndex()
        preds = json.loads(Path(pred_json).read_text(encoding="utf-8"))

    # Nếu mô hình dự đoán 0 box, trả về toàn 0 an toàn thay vì lỗi pycocotools
    cat_ids = sorted(coco_gt.getCatIds())
    if not preds:
        return {
            "mAP@0.5:0.95": 0.0,
            "mAP@0.5": 0.0,
            "mAP@0.75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AR@100": 0.0,
            "num_preds": 0,
            "per_class_ap50": {CLASS_NAMES_VI[c]: 0.0 for c in cat_ids},
            "per_class_ap": {CLASS_NAMES_VI[c]: 0.0 for c in cat_ids},
        }

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(preds)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        if img_ids is not None:
            coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats  # 12 chỉ số chuẩn COCO
    # precision có shape [T=10 IoU, R=101 Recall, K=7 Classes, A=4 Areas, M=3 MaxDets]
    precisions = coco_eval.eval["precision"]

    per_class_ap = {}
    per_class_ap50 = {}
    for k, cid in enumerate(cat_ids):
        name = CLASS_NAMES_VI.get(cid, str(cid))
        # AP@0.5:0.95 (area=all -> 0, maxDets=100 -> 2)
        p_all = precisions[:, :, k, 0, 2]
        valid_all = p_all[p_all > -1]
        per_class_ap[name] = float(np.mean(valid_all)) if valid_all.size else 0.0

        # AP@0.5 (t=0 là IoU=0.50)
        p_50 = precisions[0, :, k, 0, 2]
        valid_50 = p_50[p_50 > -1]
        per_class_ap50[name] = float(np.mean(valid_50)) if valid_50.size else 0.0

    def clean(val: float) -> float:
        return round(max(0.0, float(val)), 4)

    return {
        "mAP@0.5:0.95": clean(stats[0]),
        "mAP@0.5": clean(stats[1]),
        "mAP@0.75": clean(stats[2]),
        "AP_small": clean(stats[3]),
        "AP_medium": clean(stats[4]),
        "AP_large": clean(stats[5]),
        "AR@100": clean(stats[8]),
        "num_preds": len(preds),
        "per_class_ap50": {k: round(v, 4) for k, v in per_class_ap50.items()},
        "per_class_ap": {k: round(v, 4) for k, v in per_class_ap.items()},
    }