import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import os
from collections import defaultdict

# Define hand and pose landmarks as per your specification
hand_landmarks = [
    'INDEX_FINGER_DIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_TIP',
    'PINKY_DIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_TIP',
    'RING_FINGER_DIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_TIP',
    'THUMB_CMC', 'THUMB_IP', 'THUMB_MCP', 'THUMB_TIP', 'WRIST'
]

HAND_IDENTIFIERS = [id + "_right" for id in hand_landmarks] + [id + "_left" for id in hand_landmarks]
POSE_IDENTIFIERS = ["RIGHT_SHOULDER", "LEFT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW"]
body_identifiers = HAND_IDENTIFIERS + POSE_IDENTIFIERS  # Total of 46 keypoints

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to find the index of the first non-zero element
def find_index(array):
    for i, num in enumerate(array):
        if num != 0:
            return i
    return -1  # Return -1 if no non-zero element is found

# Function to fill in missing keypoints
def curl_skeleton(array):
    array = list(array)
    if sum(array) == 0:
        return array
    for i, location in enumerate(array):
        if location != 0:
            continue
        else:
            if i == 0 or i == len(array) - 1:
                continue
            else:
                if array[i + 1] != 0:
                    array[i] = float((array[i - 1] + array[i + 1]) / 2)
                else:
                    j = find_index(array[i + 1:])
                    if j == -1:
                        continue
                    array[i] = float(((1 + j) * array[i - 1] + array[i + 1 + j]) / (2 + j))
    return array

def process_video(video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    mp_holistic_instance = mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Prepare a dictionary to store keypoints
    keypoint_data = defaultdict(list)
    frame_count = 0

    with mp_holistic_instance as holistic:
        while frame_count < 100:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Process right hand
            if results.right_hand_landmarks:
                for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                    keypoint_data[f"{hand_landmarks[idx]}_right_x"].append(landmark.x)
                    keypoint_data[f"{hand_landmarks[idx]}_right_y"].append(landmark.y)
            else:
                for idx in range(len(hand_landmarks)):
                    keypoint_data[f"{hand_landmarks[idx]}_right_x"].append(0)
                    keypoint_data[f"{hand_landmarks[idx]}_right_y"].append(0)

            # Process left hand
            if results.left_hand_landmarks:
                for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                    keypoint_data[f"{hand_landmarks[idx]}_left_x"].append(landmark.x)
                    keypoint_data[f"{hand_landmarks[idx]}_left_y"].append(landmark.y)
            else:
                for idx in range(len(hand_landmarks)):
                    keypoint_data[f"{hand_landmarks[idx]}_left_x"].append(0)
                    keypoint_data[f"{hand_landmarks[idx]}_left_y"].append(0)

            # Process pose landmarks (shoulders and elbows)
            if results.pose_landmarks:
                landmark_dict = {mp_holistic.PoseLandmark(idx).name: idx for idx in range(len(mp_holistic.PoseLandmark))}
                for pose_identifier in POSE_IDENTIFIERS:
                    idx_pose = getattr(mp_holistic.PoseLandmark, pose_identifier).value
                    if idx_pose is not None:
                        landmark = results.pose_landmarks.landmark[idx_pose]
                        keypoint_data[f"{pose_identifier}_x"].append(landmark.x)
                        keypoint_data[f"{pose_identifier}_y"].append(landmark.y)
                    else:
                        keypoint_data[f"{pose_identifier}_x"].append(0)
                        keypoint_data[f"{pose_identifier}_y"].append(0)
            else:
                for pose_identifier in POSE_IDENTIFIERS:
                    keypoint_data[f"{pose_identifier}_x"].append(0)
                    keypoint_data[f"{pose_identifier}_y"].append(0)

    # Process the keypoints
    T = frame_count  # Number of frames processed
    num_keypoints = len(body_identifiers)
    keypoints_all_frames = np.empty((T, num_keypoints, 2))

    for index, identifier in enumerate(body_identifiers):
        x_key = identifier + "_x"
        y_key = identifier + "_y"
        x_array = keypoint_data.get(x_key, [0]*T)
        y_array = keypoint_data.get(y_key, [0]*T)
        data_keypoint_preprocess_x = curl_skeleton(x_array)
        data_keypoint_preprocess_y = curl_skeleton(y_array)
        keypoints_all_frames[:, index, 0] = np.asarray(data_keypoint_preprocess_x)
        keypoints_all_frames[:, index, 1] = np.asarray(data_keypoint_preprocess_y)

    cap.release()

    # Re-open the video to read frames again
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    os.makedirs(save_dir, exist_ok=True)
    while frame_idx < T:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = keypoints_all_frames[frame_idx]

        # Reconstruct the landmarks
        left_hand_landmarks_list = []
        right_hand_landmarks_list = []
        pose_landmarks_list = []

        # Left hand
        for i in range(len(hand_landmarks)):
            x = keypoints[i + len(hand_landmarks), 0]
            y = keypoints[i + len(hand_landmarks), 1]
            left_hand_landmarks_list.append(
                landmark_pb2.NormalizedLandmark(x=x, y=y, z=0))

        # Right hand
        for i in range(len(hand_landmarks)):
            x = keypoints[i, 0]
            y = keypoints[i, 1]
            right_hand_landmarks_list.append(
                landmark_pb2.NormalizedLandmark(x=x, y=y, z=0))

        # Pose landmarks
        for i in range(len(body_identifiers)):
            x = keypoints[i, 0]
            y = keypoints[i, 1]
            pose_landmarks_list.append(
                landmark_pb2.NormalizedLandmark(x=x, y=y, z=0))

        # Create LandmarkList objects
        left_hand_landmarks = landmark_pb2.NormalizedLandmarkList(
            landmark=left_hand_landmarks_list)
        right_hand_landmarks = landmark_pb2.NormalizedLandmarkList(
            landmark=right_hand_landmarks_list)
        pose_landmarks = landmark_pb2.NormalizedLandmarkList(
            landmark=pose_landmarks_list)

        # Draw landmarks on the frame
        mp_drawing.draw_landmarks(
            frame,
            left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        mp_drawing.draw_landmarks(
            frame,
            right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        # Draw pose landmarks (custom connections)
        # Since we're only using shoulders and elbows, we'll define custom connections
        pose_connections = [
            (42, 45),
            (43, 44),
            (42, 43),
            (40, 44),
            (20, 45)
        ]

        mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            pose_connections,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))

        # Save the frame with keypoints drawn
        output_file = os.path.join(save_dir, f"frame_{frame_idx:05d}.png")
        cv2.imwrite(output_file, frame)

        frame_idx += 1

    cap.release()

if __name__ == "__main__":
    video_path = "vsl/videos/01_Co-Hien_1-100_1-2-3_0108___center_device02_signer01_center_ord1_15.mp4"  # Replace with your video path
    save_directory = "vsl/handkp"  # Replace with your desired save directory
    process_video(video_path, save_directory)
    print("Processing completed.")
    # Kiểm tra chỉ số của 'WRIST_right'
    # idx_wrist_right = body_identifiers.index('WRIST_right')
    # print(f"Chỉ số của 'WRIST_right' là: {idx_wrist_right}")
    #
    # # Kiểm tra chỉ số của 'WRIST_left'
    # idx_wrist_left = body_identifiers.index('WRIST_left')
    # print(f"Chỉ số của 'WRIST_left' là: {idx_wrist_left}")

