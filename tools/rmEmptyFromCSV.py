import pandas as pd

# Đọc file txt để lấy danh sách các tên file cần xóa
with open('/home/ibmelab/Documents/GG/VSLRecognition/vsl/emptyFoldersPoses.txt', 'r') as file:
    # Tạo một set chứa tên file đã thêm đuôi .mp4
    delete_files = set(line.strip().split('/')[-1] + '.mp4' for line in file.readlines())

# Đường dẫn tới các file CSV
csv_files = [
    '/home/ibmelab/Documents/GG/VSLRecognition/vsl/label/labelCenter/train_labels.csv',
    '/home/ibmelab/Documents/GG/VSLRecognition/vsl/label/labelCenter/val_labels.csv',
    '/home/ibmelab/Documents/GG/VSLRecognition/vsl/label/labelCenter/test_labels.csv'
]

# Tên cột cho DataFrame, giả sử 'file_name' cho tên file và 'label_id' cho ID nhãn
column_names = ['file_name', 'label_id']

# Xử lý từng file CSV
for csv_file in csv_files:
    # Đọc dữ liệu CSV không có header và đặt tên cột
    data = pd.read_csv(csv_file, header=None, names=column_names)
    
    # Lọc dữ liệu, chỉ giữ lại những dòng không có tên file cần xóa
    filtered_data = data[~data['file_name'].isin(delete_files)]
    
    # Ghi đè lên file CSV cũ hoặc tạo file CSV mới
    filtered_data.to_csv(csv_file, index=False)
    
    print(f'Updated {csv_file}')
