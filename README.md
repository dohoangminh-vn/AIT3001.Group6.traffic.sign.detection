# Traffic sign detection — YOLO26 baseline

Repo train tối giản cho ZA Traffic 2020: Hydra quản lý cấu hình, Ultralytics quản lý vòng lặp train, W&B lưu metrics và checkpoints. Python 3.10–3.13; các lệnh bên dưới chạy từ thư mục repo.

```text
configs/config.yaml          # data, seed, output, W&B
configs/model/yolo26.yaml    # model và tham số train
src/data.py                 # đọc CSV, split theo ảnh, chuyển nhãn
src/yolo.py                 # Ultralytics + checkpoint/resume
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

CSV là nguồn annotation duy nhất. Có 4.500 ảnh, 11.000 box, 7 lớp. Chia ngẫu nhiên theo **image ID**, 3.600 train / 900 validation, seed 42; không nhóm theo `street_id`. Mọi box của một ảnh cùng tập. Public test không có nhãn nên không tính mAP.

Ảnh gốc được giữ nguyên, không crop hoặc resize trước. Ultralytics tự resize/padding và áp dụng augmentation mặc định trong lúc train. Nhãn COCO `[x,y,w,h]` đổi sang YOLO normalized; category `1–7` đổi thành `0–6`.

Lần chạy đầu tạo `data/za_traffic_yolo/` gồm ảnh symlink, labels, `data.yaml`, `split.json` (image IDs) và `train.txt`/`val.txt` (tên ảnh tương đối với thư mục ảnh gốc). Các lần sau tái sử dụng. Kaggle hỗ trợ symlink; Windows thiếu quyền symlink sẽ copy ảnh và báo cảnh báo. Không ghi vào dữ liệu nguồn.

Đổi CSV, seed hoặc tỷ lệ split thì chọn `data.prepared_dir` mới. Nếu quá trình chuẩn bị bị ngắt, xóa riêng thư mục prepared chưa hoàn tất rồi chạy lại. Muốn chỉ chuẩn bị dữ liệu:

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

**3. Smoke test trước**

Sửa `data.root` theo mount thật. Lệnh này train khoảng 2% tập train trong 2 epoch; validation vẫn dùng đủ 900 ảnh. Dùng checkpoint mỗi epoch để kiểm tra upload:

```python
!python train.py data.root=/kaggle/input/<dataset>/za_traffic_2020 model.epochs=2 model.fraction=0.02 tracking.checkpoint_every=1
```

**4. Train baseline đầy đủ**

```python
!python train.py data.root=/kaggle/input/<dataset>/za_traffic_2020
```

Mặc định: pretrained `yolo26s.pt`, `imgsz=640`, batch 4, 100 epoch, patience 30, AMP bật, workers 2, seed 42. Giữ optimizer/augmentation mặc định của Ultralytics 8.4.204. Nếu OOM, thử `model.batch_size=2` trước khi giảm `model.imgsz=512`. Một lần train mới tạo thư mục riêng trong `outputs/`.

Hydra cho phép override trực tiếp cả các tham số train của Ultralytics. Các tham số trong `model.train_args` mặc định theo recipe Ultralytics; dùng `null` để bỏ truyền một tham số và dùng giá trị mặc định của phiên bản thư viện:

```bash
python train.py model=yolo26 model.batch_size=2 model.imgsz=640
python train.py model.train_args.optimizer=AdamW model.train_args.lr0=0.001 model.train_args.lrf=0.01
python train.py model.train_args.cos_lr=true model.train_args.warmup_epochs=5 model.train_args.weight_decay=0.0001
python train.py model.train_args.mosaic=0.5 model.train_args.fliplr=0.25
python train.py tracking.project=my-traffic-project tracking.entity=my-team
python train.py --cfg job
```

Có thể cấu hình thêm các giá trị Ultralytics như `momentum`, `warmup_bias_lr`, `close_mosaic`, `hsv_h`, `degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`, `mixup` và `copy_paste` theo cùng cú pháp `model.train_args.<tên>=<giá_trị>`. Xem `configs/model/yolo26.yaml` để biết danh sách mặc định đang được truyền vào train mới.

## Xem lỗi dự đoán

Sau khi train, chạy chẩn đoán trên toàn bộ validation set để xem trực quan các ảnh có lỗi. Dùng `best.pt` của run:

```bash
python diagnose.py diagnostic.checkpoint=outputs/<run>/weights/best.pt data.root=/path/to/za_traffic_2020
```

Cũng có thể dùng checkpoint artifact trên W&B; artifact cần có `best.pt` và môi trường cần `WANDB_API_KEY`:

```bash
python diagnose.py "diagnostic.checkpoint=wandb://<entity>/<project>/run_<run_id>_checkpoint:latest" data.root=/path/to/za_traffic_2020
```

Mặc định confidence là `0.25`, ngưỡng IoU để ghép prediction với nhãn là `0.5`; có thể đổi bằng `diagnostic.conf=...` và `diagnostic.match_iou=...`. `match_iou` dùng để phân loại lỗi trong báo cáo, không thay đổi ngưỡng NMS của YOLO và không phải phép tính mAP.

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
