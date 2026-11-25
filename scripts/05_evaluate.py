import argparse
import json
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.coco_eval import analyze_detection_errors, evaluate_coco_predictions
from src.data_utils import CLASS_NAMES_VI


def save_error_grid(gt_dict: dict, worst_images: list, img_dir: Path, out_img: Path, max_samples: int = 4) -> None:
    chosen = worst_images[:max_samples]
    if not chosen:
        return
    img_map = {im["id"]: im for im in gt_dict["images"]}
    fig, axes = plt.subplots(len(chosen), 1, figsize=(14, 4.4 * len(chosen)))
    axes = [axes] if len(chosen) == 1 else axes
    for ax, (_, iid, img_tp, img_miscls, img_fp, img_fn) in zip(axes, chosen):
        ax.imshow(Image.open(img_dir / img_map[iid]["file_name"]).convert("RGB"))
        for g in img_fn:
            ax.add_patch(patches.Rectangle((g["bbox"][0], g["bbox"][1]), g["bbox"][2], g["bbox"][3], lw=2, ec="lime", fc="none"))
        for p in img_fp:
            ax.add_patch(patches.Rectangle((p["bbox"][0], p["bbox"][1]), p["bbox"][2], p["bbox"][3], lw=2, ec="red", ls="--", fc="none"))
        for g, p in img_miscls:
            ax.add_patch(patches.Rectangle((p["bbox"][0], p["bbox"][1]), p["bbox"][2], p["bbox"][3], lw=2.2, ec="magenta", fc="none"))
            ax.text(p["bbox"][0], max(0, p["bbox"][1] - 4), f"{CLASS_NAMES_VI[p['category_id']]}!={CLASS_NAMES_VI[g['category_id']]}", color="magenta", fontsize=8, bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"))
        for p in img_tp:
            ax.add_patch(patches.Rectangle((p["bbox"][0], p["bbox"][1]), p["bbox"][2], p["bbox"][3], lw=1.8, ec="cyan", fc="none"))
        ax.set_title(f"{img_map[iid]['file_name']} | TP(lơ)={len(img_tp)} | Nhầm lớp(tím)={len(img_miscls)} | Báo giả(đỏ)={len(img_fp)} | Bỏ sót(lá)={len(img_fn)}", fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.tight_layout(); plt.savefig(out_img, dpi=150); plt.close()


def plot_confusion_matrix(cm: list, name: str, out_img: Path) -> None:
    arr = np.array(cm, dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    norm = np.divide(arr, np.maximum(row_sums, 1.0))
    labels = ["Nền (BG)"] + list(CLASS_NAMES_VI.values())
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8); ax.set_yticklabels(labels, fontsize=8)
    for i in range(8):
        for j in range(8):
            cnt = int(arr[i, j])
            if cnt > 0:
                ax.text(j, i, f"{cnt}\n({norm[i,j]*100:.0f}%)", ha="center", va="center", fontsize=7, color="white" if norm[i, j] > 0.55 else "black")
    ax.set(xlabel="Dự đoán", ylabel="Nhãn thật", title=f"Ma trận nhầm lẫn 8×8 — {name}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(out_img, dpi=180); plt.close()


def plot_per_class_and_pr(results_map: dict, out_dir: Path) -> None:
    classes = list(CLASS_NAMES_VI.values())
    x = np.arange(len(classes))
    width = 0.75 / max(1, len(results_map))
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, res) in enumerate(results_map.items()):
        ax.bar(x + (i - (len(results_map) - 1) / 2) * width, [res["per_class_ap50"].get(c, 0.0) for c in classes], width, label=name)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set(ylim=(0, 1.08), ylabel="AP@0.5", title="So sánh AP@0.5 từng lớp (chuẩn pycocotools)")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(out_dir / "per_class_ap50.png", dpi=200); plt.close()

    # PR Curves cho mô hình thực cuối cùng (hoặc nhiều mô hình)
    real_models = [(n, r) for n, r in results_map.items() if n != "Oracle_SanityCheck"]
    if real_models:
        recalls = np.linspace(0, 1, 101)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        name, res = real_models[-1]
        for c in classes:
            ax.plot(recalls, res["pr_curves"].get(c, [0.0] * 101), lw=1.8, label=f"{c} (AP50={res['per_class_ap50'].get(c,0):.2f})")
        ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Recall", ylabel="Precision", title=f"Precision-Recall Curve @ IoU=0.50 ({name})")
        ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.35)
        plt.tight_layout(); plt.savefig(out_dir / "pr_curves.png", dpi=200); plt.close()


def main():
    parser = argparse.ArgumentParser(description="Đánh giá công bằng pycocotools & phân tích lỗi 4 nhóm")
    parser.add_argument("--gt", type=Path, default=Path("data/processed/coco/val.json"))
    parser.add_argument("--img-dir", type=Path, default=Path("data/za_traffic_2020/traffic_train/images"))
    parser.add_argument("--preds", nargs="*", default=[], help="Danh sách ten=duong_dan_json")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--check-oracle", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/eval"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt_data = json.loads(args.gt.read_text(encoding="utf-8"))
    pred_items = [(n, Path(p)) for item in args.preds for n, p in [item.split("=", 1)]]

    if args.check_oracle:
        oracle_path = args.out_dir / "_oracle_check.json"
        oracle_path.write_text(json.dumps([{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 1.0} for a in gt_data["annotations"]]), encoding="utf-8")
        pred_items.insert(0, ("Oracle_SanityCheck", oracle_path))

    if not pred_items:
        print("Hãy truyền ít nhất 1 file qua --preds ten=path.json hoặc --check-oracle")
        return

    rows, results_map = [], {}
    for name, ppath in pred_items:
        res = evaluate_coco_predictions(args.gt, ppath)
        preds = json.loads(ppath.read_text(encoding="utf-8"))
        err = analyze_detection_errors(gt_data, preds, conf_thresh=args.conf)
        res["error_breakdown"] = {k: err[k] for k in ("TP", "Misclassification", "FP_background", "FN", "FN_small", "Precision", "Recall", "F1")}
        results_map[name] = res
        rows.append({
            "Model": name, "mAP@0.5": res["mAP@0.5"], "mAP@0.5:0.95": res["mAP@0.5:0.95"],
            "AP_small": res["AP_small"], "AP_medium": res["AP_medium"], "AP_large": res["AP_large"],
            "P@0.25": err["Precision"], "R@0.25": err["Recall"], "F1": err["F1"],
            "TP": err["TP"], "Miscls": err["Misclassification"], "FP_bg": err["FP_background"], "FN_small": err["FN_small"],
        })
        if name != "Oracle_SanityCheck":
            save_error_grid(gt_data, err["worst_images"], args.img_dir, args.out_dir / f"{name}_errors.png")
            plot_confusion_matrix(err["confusion_matrix"], name, args.out_dir / f"{name}_confusion_matrix.png")

    df = pd.DataFrame(rows)
    print("\n=== BẢNG SO SÁNH CHUẨN COCO & PHÂN TÍCH LỖI 4 NHÓM ===")
    print(df.to_string(index=False))
    df.to_csv(args.out_dir / "comparison_table.csv", index=False)
    (args.out_dir / "detailed_metrics.json").write_text(json.dumps(results_map, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_per_class_and_pr(results_map, args.out_dir)


if __name__ == "__main__":
    main()
