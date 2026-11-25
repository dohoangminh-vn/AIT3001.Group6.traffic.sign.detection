import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.data_utils import CLASS_NAMES_VI, compute_iou_xywh


def evaluate_coco_predictions(
    gt_json: Path,
    pred_json: Path,
    img_ids: Optional[List[int]] = None,
) -> Dict:
    """Chấm điểm COCO JSON bằng pycocotools (mAP, AP_small/med/large, per-class AP & PR curves)."""
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = json.loads(Path(gt_json).read_text(encoding="utf-8"))
        coco_gt.createIndex()
        preds = json.loads(Path(pred_json).read_text(encoding="utf-8"))

    cat_ids = sorted(coco_gt.getCatIds())
    if not preds:
        return {
            "mAP@0.5:0.95": 0.0, "mAP@0.5": 0.0, "mAP@0.75": 0.0,
            "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0, "AR@100": 0.0,
            "num_preds": 0,
            "per_class_ap50": {CLASS_NAMES_VI[c]: 0.0 for c in cat_ids},
            "per_class_ap": {CLASS_NAMES_VI[c]: 0.0 for c in cat_ids},
            "pr_curves": {CLASS_NAMES_VI[c]: [0.0] * 101 for c in cat_ids},
        }

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(preds)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        if img_ids is not None:
            coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats
    precisions = coco_eval.eval["precision"]  # [T=10, R=101, K=7, A=4, M=3]
    per_class_ap, per_class_ap50, pr_curves = {}, {}, {}

    for k, cid in enumerate(cat_ids):
        name = CLASS_NAMES_VI.get(cid, str(cid))
        p_all = precisions[:, :, k, 0, 2]
        valid_all = p_all[p_all > -1]
        per_class_ap[name] = round(float(np.mean(valid_all)) if valid_all.size else 0.0, 4)

        p_50 = precisions[0, :, k, 0, 2]
        valid_50 = p_50[p_50 > -1]
        per_class_ap50[name] = round(float(np.mean(valid_50)) if valid_50.size else 0.0, 4)
        pr_curves[name] = [round(max(0.0, float(v)), 4) for v in p_50]

    clean = lambda val: round(max(0.0, float(val)), 4)
    return {
        "mAP@0.5:0.95": clean(stats[0]), "mAP@0.5": clean(stats[1]), "mAP@0.75": clean(stats[2]),
        "AP_small": clean(stats[3]), "AP_medium": clean(stats[4]), "AP_large": clean(stats[5]),
        "AR@100": clean(stats[8]), "num_preds": len(preds),
        "per_class_ap50": per_class_ap50, "per_class_ap": per_class_ap, "pr_curves": pr_curves,
    }


def analyze_detection_errors(
    gt_dict: Dict,
    preds: List[Dict],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
) -> Dict:
    """Tách lỗi 4 nhóm (TP, Misclassification, FP_background, FN_small) & Confusion Matrix 8x8."""
    gt_by_img, pred_by_img = defaultdict(list), defaultdict(list)
    for a in gt_dict["annotations"]:
        gt_by_img[a["image_id"]].append(a)
    for p in preds:
        if p["score"] >= conf_thresh:
            pred_by_img[p["image_id"]].append(p)

    cm = np.zeros((8, 8), dtype=int)  # 0: Background, 1..7: 7 lớp biển báo
    tp = miscls = fp_bg = fn = fn_small = 0
    worst_images = []

    for im in gt_dict["images"]:
        iid = im["id"]
        gts = gt_by_img[iid]
        ps = sorted(pred_by_img[iid], key=lambda x: -x["score"])
        used_g, used_p = set(), set()
        img_tp, img_miscls, img_fp = [], [], []

        def match_pass(same_class: bool):
            cands = []
            for gi, g in enumerate(gts):
                if gi in used_g:
                    continue
                for pi, p in enumerate(ps):
                    if pi in used_p:
                        continue
                    if (g["category_id"] == p["category_id"]) != same_class:
                        continue
                    iou = compute_iou_xywh(g["bbox"], p["bbox"])
                    if iou >= iou_thresh:
                        cands.append((iou, p["score"], gi, pi))
            cands.sort(key=lambda x: (-x[0], -x[1]))
            out = []
            for _, _, gi, pi in cands:
                if gi not in used_g and pi not in used_p:
                    used_g.add(gi); used_p.add(pi)
                    out.append((gts[gi], ps[pi]))
            return out

        for g, p in match_pass(same_class=True):
            cm[g["category_id"], g["category_id"]] += 1
            img_tp.append(p)
        for g, p in match_pass(same_class=False):
            cm[g["category_id"], p["category_id"]] += 1
            img_miscls.append((g, p))
        for pi, p in enumerate(ps):
            if pi not in used_p:
                cm[0, p["category_id"]] += 1
                img_fp.append(p)
        img_fn = [g for gi, g in enumerate(gts) if gi not in used_g]
        for g in img_fn:
            cm[g["category_id"], 0] += 1

        tp += len(img_tp)
        miscls += len(img_miscls)
        fp_bg += len(img_fp)
        fn += len(img_fn)
        fn_small += sum(1 for g in img_fn if g["bbox"][2] * g["bbox"][3] < 32 * 32)
        err_cnt = len(img_miscls) + len(img_fp) + len(img_fn)
        if err_cnt > 0:
            worst_images.append((err_cnt, iid, img_tp, img_miscls, img_fp, img_fn))

    worst_images.sort(key=lambda x: -x[0])
    prec = tp / max(1, tp + miscls + fp_bg)
    rec = tp / max(1, tp + miscls + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {
        "TP": tp, "Misclassification": miscls, "FP_background": fp_bg,
        "FN": fn, "FN_small": fn_small,
        "Precision": round(prec, 4), "Recall": round(rec, 4), "F1": round(f1, 4),
        "confusion_matrix": cm.tolist(), "worst_images": worst_images,
    }
