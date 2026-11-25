# Phát hiện Biển báo Giao thông Việt Nam (AIT3001 — Nhóm 06)

Pipeline tối giản, tái tạo được 100% bảng kết quả và chia sẻ chung tập dữ liệu sạch (`split_street_id.json`, 0% Data Leakage) cho cả **YOLOv8 / YOLO26** và **Faster R-CNN**.

## 1. Cấu trúc thư mục
```text
configs/
  base.yaml               # Cấu hình chung (seed, split street_id, eval, W&B/TensorBoard)
  frcnn_baseline.yaml     # Cấu hình Faster R-CNN (chạy F0..F5 bằng CLI override)
  yolo_baseline.yaml      # Cấu hình YOLOv8 / YOLO26
src/
  data_utils.py           # Load config _base_, lọc trùng 2 tầng, split street_id & manifest
  tiling.py               # Cắt 3 lát ngang 640x640 (ov 23.3%) & ghép hộp WBF / NMS
  frcnn_dataset.py        # Dataset PyTorch, On-the-fly Tiling (15% hard-neg) & Copy-Paste lớp hiếm
  frcnn_model.py          # Small Sign Anchors, RoIAlign 14x14, FocalRoIHeads (CB-Focal + GIoU)
  coco_eval.py            # Đánh giá pycocotools + tách lỗi 4 nhóm (TP/Miscls/FP_bg/FN_small)
scripts/
  01_eda.py               # EDA phân bố lớp, kích thước, IoU nội ảnh & bản đồ 3 lát cắt
  02_prepare_data.py      # Làm sạch & xuất COCO, YOLO và split_street_id.json
  03_train_yolo.py        # Huấn luyện YOLOv8/YOLO26 & xuất COCO JSON
  04_train_frcnn.py       # Huấn luyện Faster R-CNN (Val mAP mỗi epoch, W&B + TB, --resume)
  05_evaluate.py          # Bảng so sánh chung, PR Curve, Confusion Matrix 8x8 & ảnh lỗi
  06_infer_sahi_demo.py   # Suy luận Batch-4 SAHI 640x640 + đo FPS
  07_analyze_and_demo.py  # Trực quan hóa FPN/RoI Grad-CAM & Video Demo FPS (.mp4)
```

## 2. Cài đặt & Chuẩn bị dữ liệu
```bash
pip install -r requirements.txt
python scripts/01_eda.py
python scripts/02_prepare_data.py
```
File `data/processed/split_street_id.json` (và `train.txt`, `val.txt`, `test.txt`) đảm bảo mọi thành viên chấm trên cùng 722 ảnh Val / 643 ảnh Test sạch, không trùng `street_id`.

## 3. Huấn luyện & Đánh giá Faster R-CNN (Chuỗi Ablation F0 → F5)
```bash
# Smoke test nhanh trên CPU
python scripts/04_train_frcnn.py --smoke

# Chuỗi Ablation trên GPU (log song song W&B project "Traffic Sign" & TensorBoard)
python scripts/04_train_frcnn.py anchor_mode=default roi_size=7 loss_type=ce scheduler=cosine out_dir=runs/frcnn/f0_default
python scripts/04_train_frcnn.py anchor_mode=small_sign_ar roi_size=7 loss_type=ce scheduler=cosine out_dir=runs/frcnn/f1_small_ar
python scripts/04_train_frcnn.py anchor_mode=small_sign_ar roi_size=7 loss_type=ce scheduler=plateau out_dir=runs/frcnn/f2_plateau
python scripts/04_train_frcnn.py
python scripts/04_train_frcnn.py tiled=true copy_paste=true min_size=800 max_size=800 out_dir=runs/frcnn/f5_tiled640

# Batch-4 SAHI, Đánh giá pycocotools, Grad-CAM & Video Demo
python scripts/06_infer_sahi_demo.py weights=runs/frcnn/f3_focal_roi14/best.pth
python scripts/05_evaluate.py --check-oracle --preds F3=outputs/preds/f3_focal_roi14_val.json F4_SAHI=outputs/preds/f3_focal_roi14_sahi_val.json
python scripts/07_analyze_and_demo.py --mode features weights=runs/frcnn/f3_focal_roi14/best.pth
python scripts/07_analyze_and_demo.py --mode video weights=runs/frcnn/f3_focal_roi14/best.pth
```

## 4. Kiểm thử đơn vị & Minh bạch mã nguồn
```bash
pytest -v
```
- **Tự cài đặt:** Lọc trùng 2 tầng + chia `street_id` (`data_utils.py`), cắt lát 3 ô ngang & WBF (`tiling.py`), Rare-Class Copy-Paste (`frcnn_dataset.py`), `FocalRoIHeads` (Class-Balanced Focal Loss + GIoU) & `RoIAlign 14×14` (`frcnn_model.py`), đánh giá lỗi 4 nhóm (`coco_eval.py`), Batch-4 SAHI, Grad-CAM & Video Demo.
- **Thư viện / Pretrained:** `torchvision` (`FasterRCNN_ResNet50_FPN_V2`), `ultralytics`, `pycocotools`, `albumentations`, `scikit-learn`, `wandb`.

