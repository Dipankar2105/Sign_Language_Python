import os
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

DATASET_PATH = 'dataset'
FEATURES_PATH = 'features'

if not os.path.exists(FEATURES_PATH):
    os.makedirs(FEATURES_PATH)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    for phrase in os.listdir(DATASET_PATH):
        phrase_path = os.path.join(DATASET_PATH, phrase)
        if not os.path.isdir(phrase_path):
            continue

        phrase_feature_path = os.path.join(FEATURES_PATH, phrase)
        os.makedirs(phrase_feature_path, exist_ok=True)

        for video_file in os.listdir(phrase_path):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue

            video_path = os.path.join(phrase_path, video_file)
            cap = cv2.VideoCapture(video_path)
            frames_keypoints = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    keypoints = []
                    for lm in hand_landmarks.landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])
                    frames_keypoints.append(keypoints)
                else:
                    frames_keypoints.append([0.0]*63)

            cap.release()

            feature_file_name = os.path.splitext(video_file)[0] + '.npy'
            feature_save_path = os.path.join(phrase_feature_path, feature_file_name)
            np.save(feature_save_path, np.array(frames_keypoints))
            print(f'Saved features: {feature_save_path}')

print('Feature extraction completed.')
