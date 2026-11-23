import json
file_path = "data/za_traffic_2020/traffic_train/train_traffic_sign_dataset.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
new_file_path = "data/za_traffic_2020/traffic_train/train_traffic_sign_dataset_vi.json"
with open(new_file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
print("Đã convert & tạo file mới: đuôi vi.json giữ định dạng tiếng việt, không có mã ASCII")