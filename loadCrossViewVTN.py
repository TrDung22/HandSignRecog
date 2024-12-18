import torch

# Giả sử bạn đã có các mô hình được định nghĩa trong các file tương ứng
from modelling.crossVTN_model import TwoStreamCrossVTN, TwoStreamCrossViewVTN  # Model 2

# Đường dẫn tới state_dict của Model 1
model1_path = 'path_to_model1_state_dict.pth'

# Đường dẫn tới state_dict của Model 2 (có thể là một state_dict mới hoặc đã có)
model2_path = 'path_to_model2_state_dict.pth'

# Tải state_dict của Model 1
model1_state_dict = torch.load(model1_path, map_location='cpu')

# Khởi tạo Model 2 và lấy state_dict của nó
model2 = TwoStreamCrossViewVTN(
    # Thêm các tham số khởi tạo phù hợp với mô hình của bạn
    num_classes=199,
    num_heads=4,
    num_layers=4,  # Ví dụ, số lớp có thể khác
    embed_size=512,
    sequence_length=16,
    cnn='rn34',
    freeze_layers=0,
    dropout=0
)

model2_state_dict = model2.state_dict()

# Khởi tạo một dict mới để cập nhật state_dict của Model 2
new_state_dict = {}

# 1. Ánh xạ Feature Extractors và Classifier trực tiếp
for key in model1_state_dict:
    if key.startswith('feature_extractor_heatmap.') or \
       key.startswith('feature_extractor_rgb.') or \
       key.startswith('classifier.fc.'):
        if key in model2_state_dict:
            new_state_dict[key] = model1_state_dict[key]
        else:
            print(f"Warning: Key {key} not found in Model 2's state_dict")

# 2. Ánh xạ Cross Attention từ Model 1 sang Model 2
# Giả sử Model 1 có cross_attention.layers.{layer}.stream1.* và stream2.*
for key in model1_state_dict:
    if key.startswith('cross_attention.layers.'):
        parts = key.split('.')
        layer_idx = parts[2]  # Ví dụ: '0', '1', ...
        stream = parts[3]  # 'stream1' hoặc 'stream2'
        rest_of_key = '.'.join(parts[4:])  # Phần còn lại sau 'stream1.' hoặc 'stream2.'

        # Xác định module trong Model 2 tương ứng
        if stream == 'stream1':
            target_modules = ['hmap_view_cross_attn', 'stream_cross_attention']
        elif stream == 'stream2':
            target_modules = ['rgb_view_cross_attn', 'stream_cross_attention']
        else:
            print(f"Unknown stream {stream} in key {key}")
            continue

        # Ánh xạ trọng số cho từng module mục tiêu
        for module in target_modules:
            new_key = f"{module}.layers.{layer_idx}.{rest_of_key}"
            if new_key in model2_state_dict:
                new_state_dict[new_key] = model1_state_dict[key]
            else:
                print(f"Warning: Key {new_key} not found in Model 2's state_dict")

# 3. Ánh xạ position_encoding.enc.weight nếu có
if 'cross_attention.position_encoding.enc.weight' in model1_state_dict:
    position_enc_weight = model1_state_dict['cross_attention.position_encoding.enc.weight']
    target_position_enc_keys = [
        'hmap_view_cross_attn.position_encoding.enc.weight',
        'rgb_view_cross_attn.position_encoding.enc.weight',
        'stream_cross_attention.position_encoding.enc.weight'
    ]
    for pos_key in target_position_enc_keys:
        if pos_key in model2_state_dict:
            new_state_dict[pos_key] = position_enc_weight
        else:
            print(f"Warning: Position Encoding Key {pos_key} not found in Model 2's state_dict")

# 4. Ánh xạ các projector nếu cần (nếu các keys có trong Model 1)
projector_keys = ['hmap_projector.weight', 'hmap_projector.bias',
                  'rgb_projector.weight', 'rgb_projector.bias']
for key in projector_keys:
    if key in model1_state_dict and key in model2_state_dict:
        new_state_dict[key] = model1_state_dict[key]
    elif key in model1_state_dict:
        print(f"Warning: Key {key} not found in Model 2's state_dict")
    # Nếu key không có trong Model 1, bỏ qua

# 5. Cập nhật state_dict mới vào Model 2
model2_state_dict.update(new_state_dict)

# 6. Tải state_dict vào Model 2 với strict=False để bỏ qua các keys không khớp
load_result = model2.load_state_dict(model2_state_dict, strict=False)

# 7. Kiểm tra missing_keys và unexpected_keys
if load_result.missing_keys:
    print("Missing keys in the loaded state_dict:")
    for key in load_result.missing_keys:
        print(f"  {key}")
else:
    print("No missing keys.")

if load_result.unexpected_keys:
    print("Unexpected keys in the loaded state_dict:")
    for key in load_result.unexpected_keys:
        print(f"  {key}")
else:
    print("No unexpected keys.")

# 8. Lưu lại state_dict đã cập nhật nếu cần
# torch.save(model2.state_dict(), 'path_to_save_model2_updated.pth')
