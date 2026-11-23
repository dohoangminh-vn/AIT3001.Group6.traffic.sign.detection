import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# 1. Đường dẫn dữ liệu gốc và chọn ảnh 1306.png (ảnh có nhiều biển báo đủ cỡ)
DATA_DIR = Path("data/za_traffic_2020/traffic_train")
IMAGE_ID = 1306
OUT_PATH = Path("outputs/ch0_pretrained_vs_gt.jpg")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 2. Đọc nhãn thật (Ground Truth) từ file JSON
coco = json.loads((DATA_DIR / "train_traffic_sign_dataset.json").read_text(encoding="utf-8"))
cat_names = {c["id"]: c["name"] for c in coco["categories"]}
img_info = next(im for im in coco["images"] if im["id"] == IMAGE_ID)
gt_anns = [a for a in coco["annotations"] if a["image_id"] == IMAGE_ID]

img_path = DATA_DIR / "images" / img_info["file_name"]
img = Image.open(img_path).convert("RGB")

# 3. Chạy thử mô hình YOLOv8n đã huấn luyện sẵn (pretrained trên tập COCO 80 lớp)
model = YOLO("yolov8n.pt")  # Lần đầu sẽ tự tải ~6MB
result = model.predict(source=str(img_path), imgsz=640, conf=0.25, verbose=False)[0]

# 4. In tóm tắt ra màn hình
print(f"Ảnh: {img_info['file_name']} | Kích thước: {img_info['width']}x{img_info['height']} px")
print(f"\n[NHÃN THẬT - Ground Truth] Có {len(gt_anns)} biển báo:")
for a in gt_anns:
    x, y, w, h = a["bbox"]
    print(f"  - {cat_names[a['category_id']]:<18} | bbox=({x}, {y}, {w}x{h} px)")

print(f"\n[PRETRAINED YOLOv8n - COCO] Đoán ra {len(result.boxes)} vật thể:")
for box in result.boxes:
    cls_id = int(box.cls.item())
    conf = float(box.conf.item())
    x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
    print(f"  - {result.names[cls_id]:<18} | conf={conf:.2f} | box=({x1},{y1},{x2},{y2})")

# 5. Vẽ so sánh: Ảnh trên = Nhãn thật (xanh lá) | Ảnh dưới = Pretrained YOLO (đỏ)
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].imshow(img)
axes[0].set_title(f"NHÃN THẬT (Ground Truth) — {len(gt_anns)} biển báo Việt Nam", fontsize=13, fontweight="bold")
for a in gt_anns:
    x, y, w, h = a["bbox"]
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none")
    axes[0].add_patch(rect)
    axes[0].text(x, max(0, y - 5), f"{cat_names[a['category_id']]} ({w}x{h})", color="lime", fontsize=8, fontweight="bold",
                 bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))
axes[0].axis("off")

axes[1].imshow(img)
axes[1].set_title(f"PRETRAINED YOLOv8n (COCO, imgsz=640) — {len(result.boxes)} vật thể", fontsize=13, fontweight="bold")
for box in result.boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    cls_id = int(box.cls.item())
    conf = float(box.conf.item())
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
    axes[1].add_patch(rect)
    axes[1].text(x1, max(0, y1 - 5), f"{result.names[cls_id]} {conf:.2f}", color="yellow", fontsize=8, fontweight="bold",
                 bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))
axes[1].axis("off")

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"\nĐã lưu ảnh so sánh tại: {OUT_PATH.resolve()}")