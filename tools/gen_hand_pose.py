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
# LeftHand : [92,111]
# RightHand: [113,132] include 10
# Pose: [0,10] include 10

# After append neck and headtop at first:

# LeftHand : [94,113] include 113
# RightHand: [115,134] include 134
# Pose: [0,10] include 10
# Neck: 17 HeadTop: 18
"""
def impute_missing_keypoints(poses):
    """Thay thế các keypoint bị thiếu bằng giá trị từ các khung hình lân cận."""
    # 1. Thu thập các keypoint bị thiếu
    missing_keypoints = defaultdict(list)  # Khung hình -> Danh sách các keypoint bị thiếu
    for i in range(poses.shape[0]):
        for kpi in range(poses.shape[1]):
            if np.count_nonzero(poses[i, kpi]) == 0:  # Keypoint bị thiếu
                missing_keypoints[i].append(kpi)
                print(f"Keypoint bị thiếu tại khung hình {i}, keypoint {kpi}")
    # 2. Điền vào các keypoint bị thiếu
    for i in missing_keypoints.keys():
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
    return poses

def gen_pose(base_url, file_name, pose_detector):
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
            pose = pose_result['predictions'][0][0]['keypoints'][92:]
            prob = pose_result['predictions'][0][0]['keypoint_scores'][92:]
            all_poses.append(pose)
            all_probs.append(prob)

        # Chuyển đổi danh sách thành mảng NumPy
        all_poses = np.array(all_poses)  # Shape: (num_frames, num_keypoints, 2)
        all_probs = np.array(all_probs)  # Shape: (num_frames, num_keypoints)

        # Áp dụng ngưỡng xác suất
        poses_threshold = np.where(all_probs[:, :, None] > 0.2, all_poses, 0)

        # Áp dụng hàm impute_missing_keypoints
        poses_imputed = impute_missing_keypoints(poses_threshold)

        # Lưu kết quả sau khi điền keypoint bị thiếu
        for idx in range(len(poses_imputed)):
            pose = poses_imputed[idx]
            prob = all_probs[idx]
            raw_pose = [[value[0], value[1], 0] for value in pose]
            pose_threshold_02 = [[value[0], value[1], 0] if prob[i] > 0.2 else [0, 0, 0] for i, value in
                                 enumerate(pose)]
            dict_data = {
                "raw_pose": raw_pose,
                "pose_threshold_02": pose_threshold_02,
                "prob": prob.tolist()
            }
            dest = os.path.join(kp_folder, video_url.replace(".mp4", "") + '_{:06d}_'.format(idx) + 'keypoints.json')

            with open(dest, 'w') as f:
                json.dump(dict_data, f)
    else:
        print('exists')


if __name__ == "__main__":
    full_data = pd.read_csv("test.csv")
    print(full_data.columns)
    pose_detector = MMPoseInferencer("rtmpose-l_8xb64-270e_coco-wholebody-256x192")

    print(full_data.shape)
    for idx, data in full_data.iterrows():
        gen_pose("videos", data['file_name'], pose_detector)
        print("Done", data['file_name'])

