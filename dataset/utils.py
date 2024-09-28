import numpy as np
import torch
import cv2
from PIL import Image

def crop_hand(frame,keypoints,WRIST_DELTA,SHOULDER_DIST_EPSILON,
              transform,clip_len,missing_wrists_left,missing_wrists_right):
    left_wrist_index = 9
    left_elbow_index = 7
    right_wrist_index = 10
    right_elbow_index = 8

    # Crop out both wrists and apply transform
    left_wrist = keypoints[0:2, left_wrist_index]
    left_elbow = keypoints[0:2, left_elbow_index]

    left_hand_center = left_wrist + WRIST_DELTA * (left_wrist - left_elbow)
    left_hand_center_x = left_hand_center[0]
    left_hand_center_y = left_hand_center[1]
    shoulder_dist = np.linalg.norm(keypoints[0:2, 5] - keypoints[0:2, 6]) * SHOULDER_DIST_EPSILON
    left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
    left_hand_xmax = min(frame.shape[1], int(left_hand_center_x + shoulder_dist // 2))
    left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
    left_hand_ymax = min(frame.shape[0], int(left_hand_center_y + shoulder_dist // 2))

    if not np.any(left_wrist) or not np.any(
                    left_elbow) or left_hand_ymax - left_hand_ymin <= 0 or left_hand_xmax - left_hand_xmin <= 0:
                # Wrist or elbow not found -> use entire frame then
                left_hand_crop = frame
                missing_wrists_left.append(clip_len) # I tried this and achived 93% on test
                
    else:
        left_hand_crop = frame[left_hand_ymin:left_hand_ymax, left_hand_xmin:left_hand_xmax, :]
    if not isinstance(left_hand_crop,np.ndarray):
        left_hand_crop = transform(left_hand_crop.numpy())
    else:
        left_hand_crop = transform(left_hand_crop)

    right_wrist = keypoints[0:2, right_wrist_index]
    right_elbow = keypoints[0:2, right_elbow_index]
    right_hand_center = right_wrist + WRIST_DELTA * (right_wrist - right_elbow)
    right_hand_center_x = right_hand_center[0]
    right_hand_center_y = right_hand_center[1]
    right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
    right_hand_xmax = min(frame.shape[1], int(right_hand_center_x + shoulder_dist // 2))
    right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
    right_hand_ymax = min(frame.shape[0], int(right_hand_center_y + shoulder_dist // 2))

    if not np.any(right_wrist) or not np.any(
                right_elbow) or right_hand_ymax - right_hand_ymin <= 0 or right_hand_xmax - right_hand_xmin <= 0:
            # Wrist or elbow not found -> use entire frame then
            right_hand_crop = frame
            missing_wrists_right.append(clip_len) # I tried this and achived 93% on test
            
    else:
        right_hand_crop = frame[right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :]
    if not isinstance(right_hand_crop,np.ndarray):
        right_hand_crop = transform(right_hand_crop.numpy())
    else:
        right_hand_crop = transform(right_hand_crop)

    crops = torch.stack((left_hand_crop, right_hand_crop), dim=0)
    # left_hand_crop = cv2.resize(left_hand_crop,(224,224))
    # right_hand_crop = cv2.resize(right_hand_crop,(224,224))
    # new_img = np.concatenate((right_hand_crop,left_hand_crop),axis = 1)
    # crops = transform(crops)

   

    return crops,missing_wrists_left,missing_wrists_right



def crop_hand_with_keypoints(frame, crop_keypoints, hand_keypoints, WRIST_DELTA, SHOULDER_DIST_EPSILON,
                             transform, clip_len, missing_wrists_left, missing_wrists_right):
    left_wrist_index = 9
    left_elbow_index = 7
    right_wrist_index = 10
    right_elbow_index = 8

    # Crop out both wrists and apply transform
    left_wrist = crop_keypoints[0:2, left_wrist_index]
    left_elbow = crop_keypoints[0:2, left_elbow_index]

    left_hand_center = left_wrist + WRIST_DELTA * (left_wrist - left_elbow)
    left_hand_center_x = left_hand_center[0]
    left_hand_center_y = left_hand_center[1]
    shoulder_dist = np.linalg.norm(crop_keypoints[0:2, 5] - crop_keypoints[0:2, 6]) * SHOULDER_DIST_EPSILON
    left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
    left_hand_xmax = min(frame.shape[1], int(left_hand_center_x + shoulder_dist // 2))
    left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
    left_hand_ymax = min(frame.shape[0], int(left_hand_center_y + shoulder_dist // 2))

    if not np.any(left_wrist) or not np.any(
            left_elbow) or left_hand_ymax - left_hand_ymin <= 0 or left_hand_xmax - left_hand_xmin <= 0:
        # Wrist or elbow not found -> use entire frame then
        left_hand_crop = frame.copy()
        missing_wrists_left.append(clip_len)
        # Sử dụng toàn bộ keypoints cho khung hình gốc
        adjusted_left_hand_kps = hand_keypoints[:21]
    else:
        left_hand_crop = frame[left_hand_ymin:left_hand_ymax, left_hand_xmin:left_hand_xmax, :].copy()
        # Điều chỉnh tọa độ keypoints theo crop
        adjusted_left_hand_kps = hand_keypoints[:21].copy()
        adjusted_left_hand_kps[:, 0] -= left_hand_xmin
        adjusted_left_hand_kps[:, 1] -= left_hand_ymin

    # Vẽ keypoints lên hình ảnh bàn tay trái
    adjusted_left_hand_kps = adjusted_left_hand_kps.astype(int)
    print(len(adjusted_left_hand_kps))
    for kp in adjusted_left_hand_kps:
        x, y = kp
        if 0 <= x < left_hand_crop.shape[1] and 0 <= y < left_hand_crop.shape[0]:
            cv2.circle(left_hand_crop, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

    # Chuyển đổi từ BGR sang RGB nếu cần thiết
    left_hand_crop = cv2.cvtColor(left_hand_crop, cv2.COLOR_BGR2RGB)
    # Chuyển đổi sang PIL Image nếu transform yêu cầu
    left_hand_crop = Image.fromarray(left_hand_crop)
    # Áp dụng transform
    left_hand_crop = transform(left_hand_crop)

    # Xử lý bàn tay phải tương tự
    right_wrist = crop_keypoints[0:2, right_wrist_index]
    right_elbow = crop_keypoints[0:2, right_elbow_index]
    right_hand_center = right_wrist + WRIST_DELTA * (right_wrist - right_elbow)
    right_hand_center_x = right_hand_center[0]
    right_hand_center_y = right_hand_center[1]
    right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
    right_hand_xmax = min(frame.shape[1], int(right_hand_center_x + shoulder_dist // 2))
    right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
    right_hand_ymax = min(frame.shape[0], int(right_hand_center_y + shoulder_dist // 2))

    if not np.any(right_wrist) or not np.any(
            right_elbow) or right_hand_ymax - right_hand_ymin <= 0 or right_hand_xmax - right_hand_xmin <= 0:
        # Wrist or elbow not found -> use entire frame then
        right_hand_crop = frame.copy()
        missing_wrists_right.append(clip_len)
        # Sử dụng toàn bộ keypoints cho khung hình gốc
        adjusted_right_hand_kps = hand_keypoints[21:]
    else:
        right_hand_crop = frame[right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :].copy()
        # Điều chỉnh tọa độ keypoints theo crop
        adjusted_right_hand_kps = hand_keypoints[21:].copy()
        adjusted_right_hand_kps[:, 0] -= right_hand_xmin
        adjusted_right_hand_kps[:, 1] -= right_hand_ymin

    # Vẽ keypoints lên hình ảnh bàn tay phải
    adjusted_right_hand_kps = adjusted_right_hand_kps.astype(int)
    print(len(adjusted_right_hand_kps))
    for kp in adjusted_right_hand_kps:
        x, y = kp
        if 0 <= x < right_hand_crop.shape[1] and 0 <= y < right_hand_crop.shape[0]:
            cv2.circle(right_hand_crop, (x, y), radius=2, color=(0, 255, 0), thickness=-1)

    # Chuyển đổi từ BGR sang RGB nếu cần thiết
    right_hand_crop = cv2.cvtColor(right_hand_crop, cv2.COLOR_BGR2RGB)
    # Chuyển đổi sang PIL Image nếu transform yêu cầu
    right_hand_crop = Image.fromarray(right_hand_crop)
    # Áp dụng transform
    right_hand_crop = transform(right_hand_crop)

    # Kết hợp hai ảnh crop
    crops = torch.stack((left_hand_crop, right_hand_crop), dim=0)

    return crops, missing_wrists_left, missing_wrists_right

def crop_hand_with_keypoints_and_lines(frame, crop_keypoints, hand_keypoints, WRIST_DELTA, SHOULDER_DIST_EPSILON,
                            transform, clip_len, missing_wrists_left, missing_wrists_right):
    left_wrist_index = 9
    left_elbow_index = 7
    right_wrist_index = 10
    right_elbow_index = 8

    # Đảm bảo rằng các chỉ số keypoints cho tay trái và tay phải bao gồm tất cả các keypoints cần thiết
    # Nếu bạn có keypoints từ 0-20 cho tay trái và 21-41 cho tay phải, hãy sử dụng cắt 0:21 và 21:42
    left_keypoints_indices = list(range(0, 21))    # 0-20 inclusive
    right_keypoints_indices = list(range(21, 42))  # 21-41 inclusive

    # Crop out both wrists and apply transform
    left_wrist = crop_keypoints[0:2, left_wrist_index]
    left_elbow = crop_keypoints[0:2, left_elbow_index]

    left_hand_center = left_wrist + WRIST_DELTA * (left_wrist - left_elbow)
    left_hand_center_x = left_hand_center[0]
    left_hand_center_y = left_hand_center[1]
    shoulder_dist = np.linalg.norm(crop_keypoints[0:2, 5] - crop_keypoints[0:2, 6]) * SHOULDER_DIST_EPSILON
    left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
    left_hand_xmax = min(frame.shape[1], int(left_hand_center_x + shoulder_dist // 2))
    left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
    left_hand_ymax = min(frame.shape[0], int(left_hand_center_y + shoulder_dist // 2))

    if not np.any(left_wrist) or not np.any(
            left_elbow) or left_hand_ymax - left_hand_ymin <= 0 or left_hand_xmax - left_hand_xmin <= 0:
        # Wrist hoặc elbow không được tìm thấy -> sử dụng toàn bộ khung hình
        left_hand_crop = frame.copy()
        missing_wrists_left.append(clip_len)
        # Sử dụng toàn bộ keypoints cho khung hình gốc
        adjusted_left_hand_kps = hand_keypoints[left_keypoints_indices]
    else:
        left_hand_crop = frame[left_hand_ymin:left_hand_ymax, left_hand_xmin:left_hand_xmax, :].copy()
        # Điều chỉnh tọa độ keypoints theo crop
        adjusted_left_hand_kps = hand_keypoints[left_keypoints_indices].copy()
        adjusted_left_hand_kps[:, 0] -= left_hand_xmin
        adjusted_left_hand_kps[:, 1] -= left_hand_ymin

    # Vẽ keypoints và nối các keypoints cho tay trái
    adjusted_left_hand_kps = adjusted_left_hand_kps.astype(int)
    
    # Định nghĩa các kết nối cho tay trái
    left_connections = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
    
    # Vẽ các đường nối
    for connection in left_connections:
        for i in range(len(connection) - 1):
            start_idx = connection[i]
            end_idx = connection[i + 1]
            if start_idx < len(adjusted_left_hand_kps) and end_idx < len(adjusted_left_hand_kps):
                x1, y1 = adjusted_left_hand_kps[start_idx]
                x2, y2 = adjusted_left_hand_kps[end_idx]
                if (0 <= x1 < left_hand_crop.shape[1] and 0 <= y1 < left_hand_crop.shape[0] and
                    0 <= x2 < left_hand_crop.shape[1] and 0 <= y2 < left_hand_crop.shape[0]):
                    cv2.line(left_hand_crop, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
    
    # Vẽ các điểm keypoints lên hình ảnh bàn tay trái
    for kp in adjusted_left_hand_kps:
        x, y = kp
        if 0 <= x < left_hand_crop.shape[1] and 0 <= y < left_hand_crop.shape[0]:
            cv2.circle(left_hand_crop, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

    # Chuyển đổi từ BGR sang RGB nếu cần thiết
    left_hand_crop = cv2.cvtColor(left_hand_crop, cv2.COLOR_BGR2RGB)
    # Chuyển đổi sang PIL Image nếu transform yêu cầu
    left_hand_crop = Image.fromarray(left_hand_crop)
    # Áp dụng transform
    left_hand_crop = transform(left_hand_crop)

    # Xử lý bàn tay phải tương tự
    right_wrist = crop_keypoints[0:2, right_wrist_index]
    right_elbow = crop_keypoints[0:2, right_elbow_index]
    right_hand_center = right_wrist + WRIST_DELTA * (right_wrist - right_elbow)
    right_hand_center_x = right_hand_center[0]
    right_hand_center_y = right_hand_center[1]
    right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
    right_hand_xmax = min(frame.shape[1], int(right_hand_center_x + shoulder_dist // 2))
    right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
    right_hand_ymax = min(frame.shape[0], int(right_hand_center_y + shoulder_dist // 2))

    if not np.any(right_wrist) or not np.any(
            right_elbow) or right_hand_ymax - right_hand_ymin <= 0 or right_hand_xmax - right_hand_xmin <= 0:
        # Wrist hoặc elbow không được tìm thấy -> sử dụng toàn bộ khung hình
        right_hand_crop = frame.copy()
        missing_wrists_right.append(clip_len)
        # Sử dụng toàn bộ keypoints cho khung hình gốc
        adjusted_right_hand_kps = hand_keypoints[right_keypoints_indices]
    else:
        right_hand_crop = frame[right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :].copy()
        # Điều chỉnh tọa độ keypoints theo crop
        adjusted_right_hand_kps = hand_keypoints[right_keypoints_indices].copy()
        adjusted_right_hand_kps[:, 0] -= right_hand_xmin
        adjusted_right_hand_kps[:, 1] -= right_hand_ymin

    # Vẽ keypoints và nối các keypoints cho tay phải
    adjusted_right_hand_kps = adjusted_right_hand_kps.astype(int)
    
    # Định nghĩa các kết nối cho tay phải
    right_connections = [
        [21, 22, 23, 24, 25],
        [21, 26, 27, 28, 29],
        [21, 30, 31, 32, 33],
        [21, 34, 35, 36, 37],
        [21, 38, 39, 40, 41]
    ]
    
    # Vì chúng ta đã cắt các keypoints từ 21-41 thành 0-20 trong adjusted_right_hand_kps,
    # chúng ta cần điều chỉnh lại các chỉ số kết nối cho phù hợp.
    # Ví dụ: keypoint 21 trong toàn bộ là keypoint 0 trong adjusted_right_hand_kps
    adjusted_right_connections = []
    for conn in right_connections:
        adjusted_conn = [kp - 21 for kp in conn]
        adjusted_right_connections.append(adjusted_conn)
    
    # Vẽ các đường nối
    for connection in adjusted_right_connections:
        for i in range(len(connection) - 1):
            start_idx = connection[i]
            end_idx = connection[i + 1]
            if start_idx < len(adjusted_right_hand_kps) and end_idx < len(adjusted_right_hand_kps):
                x1, y1 = adjusted_right_hand_kps[start_idx]
                x2, y2 = adjusted_right_hand_kps[end_idx]
                if (0 <= x1 < right_hand_crop.shape[1] and 0 <= y1 < right_hand_crop.shape[0] and
                    0 <= x2 < right_hand_crop.shape[1] and 0 <= y2 < right_hand_crop.shape[0]):
                    cv2.line(right_hand_crop, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
    
    # Vẽ các điểm keypoints lên hình ảnh bàn tay phải
    for kp in adjusted_right_hand_kps:
        x, y = kp
        if 0 <= x < right_hand_crop.shape[1] and 0 <= y < right_hand_crop.shape[0]:
            cv2.circle(right_hand_crop, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

    # Chuyển đổi từ BGR sang RGB nếu cần thiết
    right_hand_crop = cv2.cvtColor(right_hand_crop, cv2.COLOR_BGR2RGB)
    # Chuyển đổi sang PIL Image nếu transform yêu cầu
    right_hand_crop = Image.fromarray(right_hand_crop)
    # Áp dụng transform
    right_hand_crop = transform(right_hand_crop)

    # Kết hợp hai ảnh crop
    crops = torch.stack((left_hand_crop, right_hand_crop), dim=0)

    return crops, missing_wrists_left, missing_wrists_right


def crop_hand_with_keypoints_and_lines_v2(frame, crop_keypoints, hand_keypoints, WRIST_DELTA, SHOULDER_DIST_EPSILON,
                             transform, clip_len, missing_wrists_left, missing_wrists_right):
    left_wrist_index = 9
    left_elbow_index = 7
    right_wrist_index = 10
    right_elbow_index = 8

    # Crop out both wrists and apply transform
    left_wrist = crop_keypoints[0:2, left_wrist_index]
    left_elbow = crop_keypoints[0:2, left_elbow_index]

    left_hand_center = left_wrist + WRIST_DELTA * (left_wrist - left_elbow)
    left_hand_center_x = left_hand_center[0]
    left_hand_center_y = left_hand_center[1]
    shoulder_dist = np.linalg.norm(crop_keypoints[0:2, 5] - crop_keypoints[0:2, 6]) * SHOULDER_DIST_EPSILON
    left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
    left_hand_xmax = min(frame.shape[1], int(left_hand_center_x + shoulder_dist // 2))
    left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
    left_hand_ymax = min(frame.shape[0], int(left_hand_center_y + shoulder_dist // 2))

    if not np.any(left_wrist) or not np.any(
            left_elbow) or left_hand_ymax - left_hand_ymin <= 0 or left_hand_xmax - left_hand_xmin <= 0:
        # Wrist or elbow not found -> use entire frame then
        left_hand_crop = frame.copy()
        missing_wrists_left.append(clip_len)
        # Sử dụng toàn bộ keypoints cho khung hình gốc
        adjusted_left_hand_kps = hand_keypoints[0:21]
    else:
        left_hand_crop = frame[left_hand_ymin:left_hand_ymax, left_hand_xmin:left_hand_xmax, :].copy()
        # Điều chỉnh tọa độ keypoints theo crop
        adjusted_left_hand_kps = hand_keypoints[0:21].copy()
        adjusted_left_hand_kps[:, 0] -= left_hand_xmin
        adjusted_left_hand_kps[:, 1] -= left_hand_ymin

    # Vẽ keypoints và kết nối lên hình ảnh bàn tay trái
    adjusted_left_hand_kps = adjusted_left_hand_kps.astype(int)
    # Danh sách các kết nối keypoints cho tay trái
    hand_connections = [
        [0,1,2,3,4],
        [0,5,6,7,8],
        [5,9,10,11,12],
        [9,13,14,15,16],
        [13,17,18,19,20],
        [0,17]
    ]
    # Vẽ keypoints
    for kp in adjusted_left_hand_kps:
        x, y = kp
        if 0 <= x < left_hand_crop.shape[1] and 0 <= y < left_hand_crop.shape[0]:
            cv2.circle(left_hand_crop, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
    # Vẽ kết nối
    for connection in hand_connections:
        for i in range(len(connection)-1):
            idx1 = connection[i]
            idx2 = connection[i+1]
            x1, y1 = adjusted_left_hand_kps[idx1]
            x2, y2 = adjusted_left_hand_kps[idx2]
            if (0 <= x1 < left_hand_crop.shape[1] and 0 <= y1 < left_hand_crop.shape[0] and
                0 <= x2 < left_hand_crop.shape[1] and 0 <= y2 < left_hand_crop.shape[0]):
                cv2.line(left_hand_crop, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

    # Chuyển đổi từ BGR sang RGB nếu cần thiết
    left_hand_crop = cv2.cvtColor(left_hand_crop, cv2.COLOR_BGR2RGB)
    # Chuyển đổi sang PIL Image nếu transform yêu cầu
    left_hand_crop = Image.fromarray(left_hand_crop)
    # Áp dụng transform
    left_hand_crop = transform(left_hand_crop)

    # Xử lý bàn tay phải tương tự
    right_wrist = crop_keypoints[0:2, right_wrist_index]
    right_elbow = crop_keypoints[0:2, right_elbow_index]
    right_hand_center = right_wrist + WRIST_DELTA * (right_wrist - right_elbow)
    right_hand_center_x = right_hand_center[0]
    right_hand_center_y = right_hand_center[1]
    right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
    right_hand_xmax = min(frame.shape[1], int(right_hand_center_x + shoulder_dist // 2))
    right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
    right_hand_ymax = min(frame.shape[0], int(right_hand_center_y + shoulder_dist // 2))

    if not np.any(right_wrist) or not np.any(
            right_elbow) or right_hand_ymax - right_hand_ymin <= 0 or right_hand_xmax - right_hand_xmin <= 0:
        # Wrist or elbow not found -> use entire frame then
        right_hand_crop = frame.copy()
        missing_wrists_right.append(clip_len)
        # Sử dụng toàn bộ keypoints cho khung hình gốc
        adjusted_right_hand_kps = hand_keypoints[21:42]
    else:
        right_hand_crop = frame[right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :].copy()
        # Điều chỉnh tọa độ keypoints theo crop
        adjusted_right_hand_kps = hand_keypoints[21:42].copy()
        adjusted_right_hand_kps[:, 0] -= right_hand_xmin
        adjusted_right_hand_kps[:, 1] -= right_hand_ymin

    # Vẽ keypoints và kết nối lên hình ảnh bàn tay phải
    adjusted_right_hand_kps = adjusted_right_hand_kps.astype(int)
    # Sử dụng cùng danh sách kết nối như tay trái vì các chỉ số đã được điều chỉnh
    for kp in adjusted_right_hand_kps:
        x, y = kp
        if 0 <= x < right_hand_crop.shape[1] and 0 <= y < right_hand_crop.shape[0]:
            cv2.circle(right_hand_crop, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
    for connection in hand_connections:
        for i in range(len(connection)-1):
            idx1 = connection[i]
            idx2 = connection[i+1]
            x1, y1 = adjusted_right_hand_kps[idx1]
            x2, y2 = adjusted_right_hand_kps[idx2]
            if (0 <= x1 < right_hand_crop.shape[1] and 0 <= y1 < right_hand_crop.shape[0] and
                0 <= x2 < right_hand_crop.shape[1] and 0 <= y2 < right_hand_crop.shape[0]):
                cv2.line(right_hand_crop, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

    # Chuyển đổi từ BGR sang RGB nếu cần thiết
    right_hand_crop = cv2.cvtColor(right_hand_crop, cv2.COLOR_BGR2RGB)
    # Chuyển đổi sang PIL Image nếu transform yêu cầu
    right_hand_crop = Image.fromarray(right_hand_crop)
    # Áp dụng transform
    right_hand_crop = transform(right_hand_crop)

    # Kết hợp hai ảnh crop
    crops = torch.stack((left_hand_crop, right_hand_crop), dim=0)

    return crops, missing_wrists_left, missing_wrists_right


def crop_optical_flow_hand(frame,keypoints,WRIST_DELTA,SHOULDER_DIST_EPSILON,resize,transform):
    left_wrist_index = 9
    left_elbow_index = 7
    right_wrist_index = 10
    right_elbow_index = 8

    # Crop out both wrists and apply transform
    left_wrist = keypoints[0:2, left_wrist_index]
    left_elbow = keypoints[0:2, left_elbow_index]

    left_hand_center = left_wrist + WRIST_DELTA * (left_wrist - left_elbow)
    left_hand_center_x = left_hand_center[0]
    left_hand_center_y = left_hand_center[1]
    shoulder_dist = np.linalg.norm(keypoints[0:2, 5] - keypoints[0:2, 6]) * SHOULDER_DIST_EPSILON
    left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
    left_hand_xmax = min(frame.shape[1], int(left_hand_center_x + shoulder_dist // 2))
    left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
    left_hand_ymax = min(frame.shape[0], int(left_hand_center_y + shoulder_dist // 2))

    if not np.any(left_wrist) or not np.any(
                    left_elbow) or left_hand_ymax - left_hand_ymin <= 0 or left_hand_xmax - left_hand_xmin <= 0:
                # Wrist or elbow not found -> use entire frame then
                left_hand_crop = frame                
    else:
        left_hand_crop = frame[left_hand_ymin:left_hand_ymax, left_hand_xmin:left_hand_xmax, :]
    if not isinstance(left_hand_crop,np.ndarray):
        left_hand_crop = left_hand_crop.numpy()
       
    left_hand_crop = resize(left_hand_crop)
    left_hand_crop = transform(left_hand_crop)

    right_wrist = keypoints[0:2, right_wrist_index]
    right_elbow = keypoints[0:2, right_elbow_index]
    right_hand_center = right_wrist + WRIST_DELTA * (right_wrist - right_elbow)
    right_hand_center_x = right_hand_center[0]
    right_hand_center_y = right_hand_center[1]
    right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
    right_hand_xmax = min(frame.shape[1], int(right_hand_center_x + shoulder_dist // 2))
    right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
    right_hand_ymax = min(frame.shape[0], int(right_hand_center_y + shoulder_dist // 2))

    if not np.any(right_wrist) or not np.any(
                right_elbow) or right_hand_ymax - right_hand_ymin <= 0 or right_hand_xmax - right_hand_xmin <= 0:
            # Wrist or elbow not found -> use entire frame then
            right_hand_crop = frame
            
    else:
        right_hand_crop = frame[right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :]
    if not isinstance(right_hand_crop,np.ndarray):
        right_hand_crop = right_hand_crop.numpy()
    
    right_hand_crop = resize(right_hand_crop)
    right_hand_crop = transform(right_hand_crop)

    crops = torch.stack((left_hand_crop, right_hand_crop), dim=0)
   

    return crops




def crop_center(frame,keypoints,WRIST_DELTA,SHOULDER_DIST_EPSILON,
              transform,clip_len,missing_center):

    left_wrist_index = 9
    left_elbow_index = 7
    right_wrist_index = 10
    right_elbow_index = 8

    # Crop out both wrists and apply transform
    left_wrist = keypoints[0:2, left_wrist_index]
    left_elbow = keypoints[0:2, left_elbow_index]

    left_hand_center = left_wrist + 0.15 * (left_wrist - left_elbow)
    left_hand_center_x = left_hand_center[0]
    left_hand_center_y = left_hand_center[1]
    shoulder_dist = np.linalg.norm(keypoints[0:2, 5] - keypoints[0:2, 6]) * 1.2
    left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
    left_hand_xmax = min(frame.shape[1], int(left_hand_center_x + shoulder_dist // 2))
    left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
    left_hand_ymax = min(frame.shape[0], int(left_hand_center_y + shoulder_dist // 2))
    

    right_wrist = keypoints[0:2, right_wrist_index]
    right_elbow = keypoints[0:2, right_elbow_index]
    right_hand_center = right_wrist + 0.15 * (right_wrist - right_elbow)
    right_hand_center_x = right_hand_center[0]
    right_hand_center_y = right_hand_center[1]
    right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
    right_hand_xmax = min(frame.shape[1], int(right_hand_center_x + shoulder_dist // 2))
    right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
    right_hand_ymax = min(frame.shape[0], int(right_hand_center_y + shoulder_dist // 2))

    xmin = min(min(right_hand_xmin,keypoints[0, 6]),min(left_hand_xmin,right_elbow[0]))
    xmax = max(max(left_hand_xmax,keypoints[0, 5]),max(right_hand_xmax,left_elbow[0]))
    
    ymin = min(max(0,min(left_wrist[1]-10,right_wrist[1]-10)),min(min(min(left_hand_ymin,right_hand_ymin),min(right_elbow[1],left_elbow[1])),min(keypoints[1, 6],keypoints[1, 5])))
    ymax = max(max(left_wrist[1]+10,right_wrist[1]+10),max(max(max(left_hand_ymax,right_hand_ymax),max(right_elbow[1],left_elbow[1])),max(keypoints[1, 6],keypoints[1, 5])))

    if int(ymax) - int(ymin) == 0 or int(xmax) - int(xmin) == 0:
        print("Missing center")
        missing_center.append(clip_len)
        frame = transform(frame)
        return frame,missing_center,True
    else:
        frame = transform(frame[int(ymin):int(ymax),int(xmin):int(xmax)])

        return frame,missing_center,False