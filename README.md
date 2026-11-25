# Traffic sign detection — YOLOv8s and Faster R-CNN baselines

The YOLOv8s baseline uses online 640×640 aspect-preserving letterbox resizing (Ultralytics loader); Faster R-CNN uses the same geometry through Albumentations. No resized dataset is written. Augmentation is disabled by default and, when enabled, is applied only to the training split. Run the Kaggle workflow in [notebooks/kaggle_pipeline.ipynb](notebooks/kaggle_pipeline.ipynb). The YOLO26 recipe remains available with `model=yolo26`.

Repo huấn luyện và so sánh YOLOv8s với Faster R-CNN trên ZA Traffic 2020: Hydra quản lý cấu hình, mỗi backend quản lý vòng lặp train, W&B lưu metrics và checkpoints. Python 3.10–3.13; các lệnh bên dưới chạy từ thư mục repo.

```text
configs/config.yaml          # data, seed, output, W&B
configs/model/yolov8.yaml    # YOLOv8s baseline
configs/model/fasterrcnn.yaml # Faster R-CNN v2 baseline
src/data.py                 # đọc CSV, split theo street, chuyển nhãn
src/preprocess.py            # phép biến đổi online và đổi tọa độ bbox
src/yolo.py                 # Ultralytics + checkpoint/resume
src/fasterrcnn.py            # Torchvision Faster R-CNN training
train.py                    # entrypoint Hydra
tests/                      # kiểm tra dữ liệu và artifact packaging
```

## Dữ liệu

Upload dataset lên Kaggle với cấu trúc (đường dẫn mount thực tế có thể khác):

```text
/kaggle/input/<dataset>/za_traffic_2020/
  traffic_train/
    annotation.csv
    images/*.png
  traffic_public_test/
    images/*.png
```

CSV là nguồn annotation duy nhất. Có 4.500 ảnh, 11.000 dòng annotation gốc, 7 lớp và 38 `street_id`. Split train/validation theo nhóm street với seed 42; toàn bộ ảnh của một street ở cùng một tập. `data.val_fraction=0.2` là mục tiêu gần đúng theo số ảnh, đồng thời thuật toán cân bằng tỷ lệ bbox mỗi class. Do street không bị chia nhỏ và có kích thước khác nhau, validation không nhất thiết đúng 20%. Số street, ảnh, tỷ lệ thực tế và bbox theo class được ghi trong `split.json`. Public test không có nhãn nên không tính mAP.

Khi prepare, annotation trùng chính xác trên cùng ảnh (cùng class và cùng tọa độ `[x,y,w,h]`) chỉ giữ một dòng trong label YOLO. CSV nguồn không bị sửa. Cùng bbox nhưng khác class được giữ để tránh tự ý xóa nhãn xung đột; số trường hợp này và số duplicate đã loại được ghi trong `split.json`. Prepared data cũ không được tái sử dụng nếu dedup policy hoặc split policy thay đổi; thư mục mặc định hiện tại là `data/za_traffic_yolo_street/`.

Ảnh gốc được giữ nguyên. YOLO resize/pad online qua loader Ultralytics; Faster R-CNN dùng Albumentations online. Cả hai giữ tỉ lệ và đệm giữa khung 640×640. Augmentation mặc định tắt; nếu bật, chỉ dataset train nhận augmentation. YOLO xử lý chuẩn hóa đầu vào trong framework, còn Faster R-CNN dùng chuẩn hóa riêng của torchvision.

Lần chạy đầu tạo `data/za_traffic_yolo_street/` gồm ảnh symlink, labels, `data.yaml`, `split.json` (image IDs và street IDs) và `train.txt`/`val.txt` (tên ảnh tương đối với thư mục ảnh gốc). Các lần sau tái sử dụng. Kaggle hỗ trợ symlink; Windows thiếu quyền symlink sẽ copy ảnh và báo cảnh báo. Không ghi vào dữ liệu nguồn.

Đổi CSV, seed, tỷ lệ split hoặc split policy thì chọn `data.prepared_dir` mới. Split theo street không tương thích prepared data cũ chia theo ảnh; mặc định mới dùng `data/za_traffic_yolo_street/`. Nếu quá trình chuẩn bị bị ngắt, xóa riêng thư mục prepared chưa hoàn tất rồi chạy lại. Muốn chỉ chuẩn bị dữ liệu:

```bash
python train.py prepare_only=true data.root=archive/za_traffic_2020
```

## Chạy trên Kaggle

Bật **Internet**, chọn GPU T4; pipeline chỉ dùng GPU 0. Tạo Kaggle Secret tên `WANDB_API_KEY` và cho notebook quyền truy cập secret đó. Dùng các cell sau:

**1. Clone và cài thư viện**

```python
%cd /kaggle/working
!git clone https://github.com/dohoangminh-vn/AIT3001.Group6.traffic.sign.detection.git
%cd /kaggle/working/AIT3001.Group6.traffic.sign.detection
%pip install -r requirements.txt
```

Các file triển khai phải được commit/push lên repo trước khi clone. Muốn chạy lại cùng phiên bản code, dùng `git checkout <commit>` sau clone. Không cài lại PyTorch/CUDA nếu bản Kaggle có sẵn đáp ứng requirements; pip chỉ thay khi dependency không tương thích. Nếu notebook đã import thư viện trước khi cài, restart kernel sau cài đặt.

**2. Đăng nhập W&B qua secret**

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

Không in API key hoặc ghi key vào config/Git.

**3. Smoke test YOLO trước**

Sửa `data.root` theo mount thật. Lệnh này train YOLO trên khoảng 2% tập train trong 2 epoch; validation vẫn dùng toàn bộ validation. Dùng checkpoint mỗi epoch để kiểm tra upload:

```python
!python train.py data.root=/kaggle/input/<dataset>/za_traffic_2020 model.epochs=2 model.fraction=0.02 tracking.checkpoint_every=1
```

**4. Train baseline đầy đủ**

```python
!python train.py data.root=/kaggle/input/<dataset>/za_traffic_2020
```

Mặc định train pretrained `yolov8s.pt`, 640×640, batch 4, tối đa 50 epoch, patience 10, AMP bật, workers 2, seed 42 và dùng toàn bộ dữ liệu train. Chọn Faster R-CNN bằng `model=fasterrcnn`; model này dùng AdamW cùng Linear Warmup và CosineAnnealingLR. Notebook Kaggle có lệnh chạy đầy đủ. Nếu OOM, giảm batch riêng theo model. Mỗi run tạo thư mục riêng trong `outputs/`.

Đặt nhãn dễ nhớ cho run mới bằng `tracking.name`; pipeline tự nối timestamp và mã ngắn để phân biệt các lần chạy cùng nhãn. Tên này được dùng cho W&B run và thư mục output. Ví dụ:

```bash
python train.py tracking.name=baseline-yolov8s
python train.py model=fasterrcnn tracking.name=baseline-fasterrcnn
python train.py augmentation.enabled=true model.augmentation.fliplr=0.25
python train.py evaluation.inference.score_threshold=0.005 evaluation.inference.nms_iou=0.65
```

Để `tracking.name=null` (mặc định), run tự tạo tên dựa trên model, timestamp và mã ngắn.

Các tham số train của YOLO vẫn có thể override trực tiếp bằng Hydra. Tham số ảnh chung nằm trong `data`, ngưỡng đánh giá trong `evaluation`, và công tắc augmentation trong `augmentation.enabled`. Mỗi run lưu cấu hình đã resolve để tái lập.

```bash
python train.py model=yolov8 model.batch_size=2 data.image_size=640
python train.py model.train_args.optimizer=AdamW model.train_args.lr0=0.001 model.train_args.lrf=0.01
python train.py model.augmentation.mosaic=0.5 model.augmentation.fliplr=0.25 augmentation.enabled=true
python train.py model=fasterrcnn model.optimizer.lr=0.00005 model.scheduler.warmup_epochs=5
python train.py tracking.project=my-traffic-project tracking.entity=my-team
python train.py --cfg job
```

Có thể cấu hình thêm các giá trị Ultralytics như `momentum`, `warmup_bias_lr`, `close_mosaic`, `hsv_h`, `degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`, `mixup` và `copy_paste` theo cùng cú pháp `model.train_args.<tên>=<giá_trị>`. Xem `configs/model/yolo26.yaml` để biết danh sách mặc định đang được truyền vào train mới.

Baseline mới resize online, giữ tỉ lệ và đệm giữa khung 640×640. YOLO dùng loader Ultralytics; Faster R-CNN dùng Albumentations.

## Xem lỗi dự đoán

Tune ngưỡng inference trên validation theo tên run W&B chính xác. Mặc định tune NMS IoU theo mAP@0.5:0.95 rồi tune confidence theo F1; chỉ ghi CSV kết quả và `best_thresholds.json` vào `outputs/tuning/`:

```bash
python tune.py tuning.run_name=<exact-wandb-run-name> data.root=/path/to/za_traffic_2020
```

Có thể chọn riêng ngưỡng/metric, ví dụ `tuning.thresholds=[conf] tuning.conf.metric=recall`. Metric hỗ trợ `map50_95`, `map50`, `f1`, `precision`, `recall`. mAP dùng confidence `0.001`; precision/recall/F1 dùng confidence cố định khi tune NMS. Diagnostics chạy riêng sau đó bằng các ngưỡng đã chọn.

Sau khi train, chạy chẩn đoán trên toàn bộ validation set để xem trực quan các ảnh có lỗi. Dùng `best.pt` của run:

```bash
python diagnose.py diagnostic.checkpoint=outputs/<run>/weights/best.pt data.root=/path/to/za_traffic_2020
```

Cũng có thể dùng checkpoint artifact trên W&B; artifact cần có `best.pt` và môi trường cần `WANDB_API_KEY`:

```bash
python diagnose.py "diagnostic.checkpoint=wandb://<entity>/<project>/run_<run_id>_checkpoint:latest" data.root=/path/to/za_traffic_2020
```

Mặc định confidence là `0.25`, ngưỡng IoU để ghép prediction với nhãn là `0.5`; có thể đổi bằng `diagnostic.conf=...` và `diagnostic.match_iou=...`. `match_iou` dùng để phân loại lỗi trong báo cáo, không thay đổi ngưỡng NMS của YOLO và không phải phép tính mAP.

Sau khi train xong, pipeline dùng `best.pt` để đánh giá một lượt trên validation theo diện tích bbox gốc `<128`, `<256`, `<512` và `<1024 px²`. Kết quả AP@0.5, AP@0.5:0.95 cùng số bbox/ảnh validation được ghi vào `outputs/<run>/small_object_metrics.csv` và log vào cùng W&B run dưới `small_eval/`. Các metric này bổ sung cho mAP tổng thể theo epoch. Evaluator dùng COCO area filtering; cần `pycocotools` theo `requirements.txt`.

Batch inference mặc định dùng cùng `model.batch_size` của cấu hình train (hiện là 4), đồng thời bật FP16 half-precision trên GPU để giảm VRAM. Nếu diagnostics thiếu VRAM, giảm riêng batch bằng `diagnostic.batch_size=1` hoặc `diagnostic.batch_size=2`; có thể tắt half precision bằng `diagnostic.amp=false` để đối chiếu kết quả. Trên CPU, FP16 tự tắt.

Mỗi lần chạy tạo `outputs/diagnostics/<checkpoint>-<timestamp>/` gồm `images/` (chỉ các ảnh có lỗi), `image_summary.csv` (một dòng cho mọi ảnh validation), `errors.csv` (một dòng cho mỗi box FP/FN hoặc nhầm lớp) và `report.json` (checkpoint, split và ngưỡng). Box đúng màu xanh lá, false positive màu đỏ, false negative màu cam, nhầm lớp màu tím. Các số đếm lỗi trong báo cáo phục vụ điều tra ảnh; một nhầm lớp tính là một sự kiện lỗi và các số này không thay cho mAP. Public test không được dùng vì không có ground truth.

## W&B và resume

Tích hợp Ultralytics ghi loss, learning rate, mAP@0.5, mAP@0.5:0.95, precision, recall và biểu đồ; tự ghi tham số train. Repo không bổ sung Git/GPU/dependency logging hay upload split manifest.

- Mỗi 5 epoch (và epoch cuối), callback upload artifact loại `checkpoint`, tên `run_<run_id>_checkpoint`, alias `latest`. Chứa `last.pt` có optimizer, `best.pt` hiện có và `run.json` để nối lại đúng W&B run/dataset/split. Sau khi upload thành công, callback xóa các version checkpoint cũ cùng run, chỉ giữ version mới nhất để resume; artifact model tốt nhất không bị xóa.
- Khi kết thúc, tích hợp Ultralytics đánh giá lại best model và upload artifact loại `model`, tên `run_<run_id>_model`, alias `best`. Tiêu chí best là validation mAP@0.5:0.95.
- Checkpoint phục hồi định kỳ khác với `weights/last.pt` và `weights/best.pt` sau khi train kết thúc: các file cuối có thể đã bị loại optimizer. Best weights dùng inference, không dùng resume.
- Do chỉ giữ checkpoint mới nhất, Kaggle ngắt đột ngột chỉ phục hồi được tới mốc đã upload gần nhất. Việc dọn version cũ cần W&B API key có quyền xóa artifact; nếu upload thành công nhưng không đủ quyền xóa, checkpoint mới vẫn có alias `latest` còn callback sẽ báo lỗi. Checkpoint epoch cuối đã hoàn tất toàn bộ budget không còn epoch để resume.

Resume từ W&B trong phiên Kaggle mới, dùng chính xác artifact URI in ở cuối run hoặc trên trang Artifacts:

```python
!python train.py data.root=/kaggle/input/<dataset>/za_traffic_2020 resume=<entity>/<project>/run_<run_id>_checkpoint:latest
```

Hoặc local: tải artifact, giữ `last.pt`, `best.pt`, `run.json` cùng thư mục:

```bash
python train.py resume=/path/to/downloaded-artifact/last.pt
```

Resume tiếp tục cùng W&B run ID và giữ epoch budget, optimizer, recipe từ checkpoint. Các giá trị `model.train_args.*` chỉ áp dụng cho run mới; khi resume chúng không được truyền vào Ultralytics. Có thể override batch, imgsz, device và workers. Không dùng resume để đổi dataset/split hoặc kéo dài run đã train xong; bắt đầu run mới cho thí nghiệm khác. Output của phiên resume vẫn nằm trong thư mục mới trên working hiện tại; best model từ phiên trước được giữ lại.

Để kiểm tra resume thật trên Kaggle: chạy 10 epoch với `tracking.checkpoint_every=1`, đợi thông báo upload epoch đầu, ngắt cell rồi resume từ artifact `latest`. Xác nhận log tiếp tục epoch kế tiếp và W&B vẫn cùng run. Giữ `model.fraction=0.02` từ đầu cho phép thử ngắn; resume giữ fraction từ checkpoint.

Có thể chủ động chạy `tracking.mode=offline` khi không có mạng: dữ liệu W&B lưu trong outputs, chưa được upload. Sau đó dùng `wandb sync <offline-run-directory>` khi có mạng. Resume cùng online run yêu cầu `tracking.mode=online`.

## Kiểm tra và thêm model

```bash
python -m unittest discover -s tests -v
```

Các test nhỏ kiểm tra bbox, class mapping, split không trùng ảnh, reuse, giữ nguồn nguyên vẹn và đóng gói/download checkpoint bằng mock. Chúng không thay thế smoke test GPU hoặc kiểm chứng optimizer resume thật trên Kaggle.

Sau này thêm `src/fasterrcnn.py` với `train(cfg)`, `configs/model/fasterrcnn.yaml` và nhánh chọn backend trong `train.py`. Dùng lại CSV và split đã lưu, thêm TorchVision dataset adapter/Lightning Trainer khi cần. Giữ resize, optimizer, batch riêng cho từng model; không ép các backend dùng cùng trainer.
