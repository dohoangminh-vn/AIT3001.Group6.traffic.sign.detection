import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.coco_eval import evaluate_coco_predictions
from src.data_utils import CLASS_NAMES_VI, compute_iou_xywh


def save_error_analysis(
    gt_json: Path,
    pred_json: Path,
    img_dir: Path,
    out_img: Path,
    conf_thresh: float = 0.25,
    max_samples: int = 4,
) -> dict:
    """
    Tự động phân tích lỗi ở ngưỡng conf_thresh:
    - Đếm TP (đúng), FP (báo động giả), FN (bỏ sót — tách riêng biển báo nhỏ < 32x32).
    - Vẽ lưới tối đa 4 ảnh có nhiều lỗi nhất (xanh lá: GT bị bỏ sót, đỏ: FP, xanh lơ: TP).
    """
    gt = json.loads(gt_json.read_text(encoding="utf-8"))
    preds = json.loads(pred_json.read_text(encoding="utf-8"))

    img_map = {im["id"]: im for im in gt["images"]}
    gt_by_img = defaultdict(list)
    for a in gt["annotations"]:
        gt_by_img[a["image_id"]].append(a)

    pred_by_img = defaultdict(list)
    for p in preds:
        if p["score"] >= conf_thresh:
            pred_by_img[p["image_id"]].append(p)

    tp_total, fp_total, fn_total, fn_small = 0, 0, 0, 0
    error_records = []

    for iid, gt_list in gt_by_img.items():
        p_list = sorted(pred_by_img[iid], key=lambda x: -x["score"])
        matched_gt = set()
        img_tp, img_fp = [], []

        for p in p_list:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gt_list):
                if j in matched_gt or g["category_id"] != p["category_id"]:
                    continue
                iou = compute_iou_xywh(p["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.5 and best_j >= 0:
                matched_gt.add(best_j)
                img_tp.append(p)
            else:
                img_fp.append(p)

        img_fn = [g for j, g in enumerate(gt_list) if j not in matched_gt]
        tp_total += len(img_tp)
        fp_total += len(img_fp)
        fn_total += len(img_fn)
        fn_small += sum(1 for g in img_fn if g["bbox"][2] * g["bbox"][3] < 32 * 32)

        if len(img_fp) + len(img_fn) > 0:
            error_records.append((len(img_fn) + len(img_fp), iid, img_tp, img_fp, img_fn))

    # Chọn các ảnh có nhiều lỗi nhất để trực quan hóa
    error_records.sort(key=lambda x: -x[0])
    chosen = error_records[:max_samples]

    if chosen:
        fig, axes = plt.subplots(len(chosen), 1, figsize=(14, 4.5 * len(chosen)))
        if len(chosen) == 1:
            axes = [axes]
        for ax, (_, iid, img_tp, img_fp, img_fn) in zip(axes, chosen):
            im = Image.open(img_dir / img_map[iid]["file_name"]).convert("RGB")
            ax.imshow(im)
            for g in img_fn:
                x, y, w, h = g["bbox"]
                ax.add_patch(patches.Rectangle((x, y), w, h, lw=2, ec="lime", fc="none"))
            for p in img_fp:
                x, y, w, h = p["bbox"]
                ax.add_patch(patches.Rectangle((x, y), w, h, lw=2, ec="red", ls="--", fc="none"))
            for p in img_tp:
                x, y, w, h = p["bbox"]
                ax.add_patch(patches.Rectangle((x, y), w, h, lw=2, ec="cyan", fc="none"))
            ax.set_title(
                f"{img_map[iid]['file_name']} | Đúng(TP xanh lơ)={len(img_tp)} | "
                f"Bỏ sót(FN xanh lá)={len(img_fn)} | Báo giả(FP đỏ)={len(img_fp)}",
                fontsize=11, fontweight="bold",
            )
            ax.axis("off")
        out_img.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_img, dpi=150)
        plt.close()

    return {"TP": tp_total, "FP": fp_total, "FN": fn_total, "FN_small": fn_small}


def plot_per_class_comparison(results_map: dict, out_img: Path):
    """Vẽ biểu đồ cột so sánh AP@0.5 của 7 lớp giữa các mô hình."""
    classes = list(CLASS_NAMES_VI.values())
    x = np.arange(len(classes))
    n_models = len(results_map)
    width = 0.75 / max(1, n_models)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, res) in enumerate(results_map.items()):
        vals = [res["per_class_ap50"].get(c, 0.0) for c in classes]
        ax.bar(x + (i - (n_models - 1) / 2) * width, vals, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("AP@0.5")
    ax.set_title("So sánh AP@0.5 trên từng lớp biển báo (chuẩn pycocotools)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_img, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Đánh giá công bằng nhiều mô hình bằng pycocotools")
    parser.add_argument("--gt", type=Path, default=Path("data/processed/coco/val.json"))
    parser.add_argument("--img-dir", type=Path, default=Path("data/za_traffic_2020/traffic_train/images"))
    parser.add_argument("--preds", nargs="*", default=[], help="Danh sách ten=duong_dan_json")
    parser.add_argument("--check-oracle", action="store_true", help="Thêm kiểm chứng chuẩn (GT tự chấm chính nó -> phải ra 1.0)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/eval"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_items = []
    for item in args.preds:
        name, path_str = item.split("=", 1)
        pred_items.append((name, Path(path_str)))

    # Tùy chọn kiểm chứng bộ chấm (Oracle: dùng chính GT làm dự đoán -> mọi chỉ số phải bằng 1.0000!)
    if args.check_oracle:
        gt_data = json.loads(args.gt.read_text(encoding="utf-8"))
        oracle_preds = [
            {"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0}
            for a in gt_data["annotations"]
        ]
        oracle_path = args.out_dir / "_oracle_check.json"
        oracle_path.write_text(json.dumps(oracle_preds), encoding="utf-8")
        pred_items.insert(0, ("Oracle_SanityCheck", oracle_path))

    if not pred_items:
        print("Hãy truyền ít nhất 1 file qua --preds ten=path.json hoặc --check-oracle")
        return

    rows = []
    results_map = {}
    for name, ppath in pred_items:
        res = evaluate_coco_predictions(args.gt, ppath)
        results_map[name] = res
        rows.append({
            "Model": name,
            "mAP@0.5": res["mAP@0.5"],
            "mAP@0.5:0.95": res["mAP@0.5:0.95"],
            "AP_small": res["AP_small"],
            "AP_medium": res["AP_medium"],
            "AP_large": res["AP_large"],
            "Preds": res["num_preds"],
        })

        # Xuất ảnh phân tích lỗi cho từng mô hình thật (bỏ qua oracle)
        if name != "Oracle_SanityCheck":
            err = save_error_analysis(args.gt, ppath, args.img_dir, args.out_dir / f"{name}_errors.png")
            print(f"[Phân tích lỗi {name} @ conf=0.25] TP={err['TP']} | FP={err['FP']} | FN={err['FN']} (trong đó biển nhỏ={err['FN_small']})")

    df = pd.DataFrame(rows)
    print("\n=== BẢNG SO SÁNH CHUẨN COCO (DÙNG CHO BÁO CÁO ACL) ===")
    print(df.to_string(index=False))

    # Lưu CSV + Markdown + JSON chi tiết từng lớp
    df.to_csv(args.out_dir / "comparison_table.csv", index=False)
    (args.out_dir / "detailed_metrics.json").write_text(
        json.dumps(results_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_per_class_comparison(results_map, args.out_dir / "per_class_ap50.png")
    print(f"\nĐã lưu bảng kết quả & biểu đồ tại: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()