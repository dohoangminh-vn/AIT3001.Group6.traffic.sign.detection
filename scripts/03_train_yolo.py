import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import CLASS_NAMES_ASCII, load_config, yolo_cls_to_coco_cat


def build_smoke_dataset(yolo_root: Path, n_images: int = 16) -> Path:
    train_imgs = sorted((yolo_root / "images" / "train").glob("*.png"))[:n_images]
    smoke_dir = Path("data/processed/yolo_smoke")
    smoke_dir.mkdir(parents=True, exist_ok=True)
    list_file = smoke_dir / "smoke_list.txt"
    list_file.write_text("\n".join(str(p.resolve()) for p in train_imgs) + "\n", encoding="utf-8")
    smoke_yaml = smoke_dir / "dataset_smoke.yaml"
    smoke_yaml.write_text(yaml.safe_dump({"train": str(list_file.resolve()), "val": str(list_file.resolve()), "names": CLASS_NAMES_ASCII}, sort_keys=False), encoding="utf-8")
    return smoke_yaml


def export_coco_predictions(weights_path: Path, coco_gt_path: Path, img_dir: Path, imgsz: int, out_json: Path, limit: int | None = None) -> None:
    coco_gt = json.loads(coco_gt_path.read_text(encoding="utf-8"))
    images = coco_gt["images"][:limit] if limit else coco_gt["images"]
    model = YOLO(str(weights_path))
    preds = []
    for im in images:
        res = model.predict(source=str(img_dir / im["file_name"]), imgsz=imgsz, conf=0.001, iou=0.65, verbose=False)[0]
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            preds.append({
                "image_id": im["id"],
                "category_id": yolo_cls_to_coco_cat(int(box.cls.item())),
                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(float(box.conf.item()), 5),
            })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(preds), encoding="utf-8")
    print(f"Đã xuất {len(preds)} dự đoán chuẩn COCO -> {out_json.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện YOLO (YOLOv8 / YOLO26)")
    parser.add_argument("--config", type=Path, default=Path("configs/yolo_baseline.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("opts", nargs="*", help="Ghi đè cấu hình dạng key=val")
    args = parser.parse_args()

    cfg = load_config(args.config, args.opts)
    if args.model: cfg["model"] = args.model
    if args.imgsz: cfg["imgsz"] = args.imgsz
    if args.epochs: cfg["epochs"] = args.epochs
    if args.name: cfg["name"] = args.name

    device = 0 if torch.cuda.is_available() else "cpu"
    if args.smoke:
        data_yaml = build_smoke_dataset(Path(cfg["data"]["yolo_dir"]), n_images=16)
        train_kwargs = dict(data=str(data_yaml), epochs=5, imgsz=640, batch=4, device=device, workers=0, fliplr=0.0, mosaic=0.0, seed=cfg["seed"], project=str(Path("runs/detect").resolve()), name="yolo_smoke", exist_ok=True, plots=True)
        model_name = "yolov8n.pt"
    else:
        train_kwargs = dict(data=str(Path(cfg["data_yaml"]).resolve()), epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"], lr0=cfg["lr0"], optimizer=cfg["optimizer"], device=device, workers=cfg["workers"], fliplr=cfg["fliplr"], flipud=cfg["flipud"], mosaic=cfg["mosaic"], hsv_h=cfg["hsv_h"], hsv_s=cfg["hsv_s"], hsv_v=cfg["hsv_v"], seed=cfg["seed"], project=cfg["project"], name=cfg["name"], exist_ok=True, plots=True)
        model_name = cfg["model"]

    model = YOLO(model_name)
    model.train(**train_kwargs)
    best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    export_coco_predictions(best_pt, Path(cfg["data"]["coco_dir"]) / "val.json", Path(cfg["data"]["yolo_dir"]) / "images/val", train_kwargs["imgsz"], Path("outputs/preds") / f"{train_kwargs['name']}_val.json", limit=16 if args.smoke else None)


if __name__ == "__main__":
    main()
