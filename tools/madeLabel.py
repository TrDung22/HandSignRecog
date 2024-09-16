import os
import csv

# Đường dẫn đến thư mục chứa video
directory = '/home/ibmelab/Documents/GG/VSLRecognition/vsl/videos'

# Tên file CSV để lưu kết quả
csv_file = '/home/ibmelab/Documents/GG/VSLRecognition/vsl/vsl_200_all.csv'

# Mở file CSV để ghi dữ liệu
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # Ghi hàng tiêu đề
    writer.writerow(['file_name', 'label_id', 'direction', 'person'])
    
    # Duyệt qua tất cả các file trong thư mục
    for filename in os.listdir(directory):
        if filename.endswith(".mp4") and 'ord1' in filename:
            parts = filename.split('_')
            if len(parts) > 9:  # Đảm bảo rằng tên file có đủ các phần
                person = parts[1]  # Lấy thông tin người
                direction_parts = filename.split('___')
                if len(direction_parts) > 1:
                    direction = direction_parts[1].split('_')[0]  # Lấy thông tin hướng
                    label_id = filename.split('ord1_')[-1].split('.')[0]  # Lấy label_id
                    # Ghi thông tin vào file CSV
                    writer.writerow([filename, label_id, direction, person])
                else:
                    print(f"Không thể phân tích 'direction' trong tên file: {filename}")
            else:
                print(f"Tên file không đúng định dạng mong đợi: {filename}")
        elif filename.endswith(".mp4"):
            print(f"File bị bỏ qua vì không phải là 'ord1': {filename}")

print("Đã ghi xong file CSV.")
