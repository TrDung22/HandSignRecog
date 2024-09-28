from mmpose.apis import MMPoseInferencer
import numpy as np
import cv2
import time
import json
import os
import pandas as pd
from tqdm.auto import tqdm
import glob
from collections import defaultdict

"""
# LeftHand : [0,20]
# RightHand: [21,41]
"""

def impute_missing_keypoints(poses, file_name):
    """Thay thế các keypoint bị thiếu bằng giá trị từ các khung hình lân cận."""
    # Danh sách để lưu trữ thông tin về các keypoint bị thiếu
    missing_keypoints_info = []

    # 1. Thu thập các keypoint bị thiếu
    missing_keypoints = defaultdict(list)  # Khung hình -> Danh sách các keypoint bị thiếu
    for i in range(poses.shape[0]):  # Duyệt qua từng khung hình
        for kpi in range(poses.shape[1]):  # Duyệt qua từng keypoint
            if np.count_nonzero(poses[i, kpi]) == 0:  # Keypoint bị thiếu
                missing_keypoints[i].append(kpi)
                # Không in ra màn hình nữa
                # Thu thập thông tin vào danh sách
                missing_keypoints_info.append({
                    'name': file_name,
                    'missing_frame': i,
                    'missing_keypoint_index': kpi,
                    'replacement_frame': None  # Sẽ cập nhật sau
                })

    # 2. Điền vào các keypoint bị thiếu
    for idx, i in enumerate(missing_keypoints.keys()):
        missing = missing_keypoints[i]
        for kpi in missing:
            # Các ứng viên thay thế
            candidates = poses[:, kpi]
            min_dist = np.inf
            replacement = -1
            for f in range(candidates.shape[0]):
                if f != i and np.count_nonzero(candidates[f]) > 0:
                    distance = abs(f - i)
                    if distance < min_dist:
                        min_dist = distance
                        replacement = f
            # Thay thế
            if replacement > -1:
                poses[i, kpi] = poses[replacement, kpi]
                # Cập nhật thông tin replacement_frame trong danh sách
                for info in missing_keypoints_info:
                    if info['name'] == file_name and info['missing_frame'] == i and info['missing_keypoint_index'] == kpi:
                        info['replacement_frame'] = replacement
            else:
                # Nếu không tìm thấy frame thay thế, có thể để giá trị mặc định hoặc xử lý khác
                pass

    return poses, missing_keypoints_info

def gen_pose(base_url, file_name, pose_detector, csv_writer):
    try:
        video_url = os.path.join(base_url, file_name)
        pose_results = pose_detector(video_url)

        # kp_folder = video_url.replace("video_200_400",'poses_200_400').replace('.mp4',"")
        kp_folder = video_url.replace("videos", 'hand_poses').replace('.mp4', "")
        if not os.path.exists(kp_folder):
            os.makedirs(kp_folder, exist_ok=True)

            # Danh sách để lưu trữ pose và prob cho tất cả các khung hình
            all_poses = []
            all_probs = []

            for idx, pose_result in enumerate(pose_results):
                pose = pose_result['predictions'][0][0]['keypoints'][91:]
                prob = pose_result['predictions'][0][0]['keypoint_scores'][91:]
                all_poses.append(pose)
                all_probs.append(prob)

            # Chuyển đổi danh sách thành mảng NumPy
            all_poses = np.array(all_poses)  # Shape: (num_frames, num_keypoints, 2)
            all_probs = np.array(all_probs)  # Shape: (num_frames, num_keypoints)

            # Áp dụng ngưỡng xác suất
        
            poses_threshold = np.where(all_probs[:, :, None] > 0.2, all_poses, 0)

            # Áp dụng hàm impute_missing_keypoints và nhận thông tin về keypoint bị thiếu
            poses_imputed, missing_keypoints_info = impute_missing_keypoints(poses_threshold, file_name)

            # Lưu thông tin keypoint bị thiếu vào CSV
            if missing_keypoints_info:
                for info in missing_keypoints_info:
                    csv_writer.writerow(info)

            # Lưu kết quả sau khi điền keypoint bị thiếu
            for idx in range(len(poses_imputed)):
                pose = poses_imputed[idx]
                prob = all_probs[idx]
                raw_pose = [[value[0], value[1], 0] for value in pose]
                pose_threshold_02 = [[value[0], value[1], 0] if prob[i] > 0.2 else [0, 0, 0] for i, value in enumerate(pose)]
                dict_data = {
                    "raw_pose": raw_pose,
                    "pose_threshold_02": pose_threshold_02,
                    "prob": prob.tolist()
                }
                dest = os.path.join(kp_folder, file_name.replace(".mp4", "") + '_{:06d}_'.format(idx) + 'keypoints.json')

                with open(dest, 'w') as f:
                    json.dump(dict_data, f)
        else:
            print('exists')
    except Exception as e:
        print(f"An error occurred with file {file_name}: {e}")


if __name__ == "__main__":
    import csv

    full_data = pd.read_csv("/home/ibmelab/Documents/GG/VSLRecognition/AUTSL/full_labels.csv")
    print(full_data.columns)
    pose_detector = MMPoseInferencer("rtmpose-m_8xb64-270e_coco-wholebody-256x192")

    # Mở file CSV để ghi thông tin về keypoint bị thiếu
    with open('/home/ibmelab/Documents/GG/VSLRecognition/AUTSL/missing_keypoints_info.csv', mode='w', newline='') as csv_file:
        fieldnames = ['name', 'missing_frame', 'missing_keypoint_index', 'replacement_frame']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Ghi tiêu đề cột
        csv_writer.writeheader()

        print(full_data.shape)
        for idx, data in full_data.iterrows():
            gen_pose("/home/ibmelab/Documents/GG/VSLRecognition/AUTSL/videos", data['name'], pose_detector, csv_writer)
            print("Done", data['name'])
