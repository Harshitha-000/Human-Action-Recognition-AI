import cv2
import mediapipe as mp
import numpy as np
import os

actions = ['eating','drinking','walking','sitting','waving']
DATA_PATH = "dataset"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
    running_mode=VisionRunningMode.VIDEO
)

landmarker = PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

timestamp = 0

for action in actions:

    print(f"Collecting data for {action}")

    for sequence in range(20):  # 20 recordings

        keypoints_sequence = []

        for frame_num in range(30):  # 30 frames

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect_for_video(mp_image, timestamp)
            timestamp += 1

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                keypoints = []
                for lm in landmarks:
                    keypoints.extend([lm.x, lm.y, lm.z])

                keypoints_sequence.append(keypoints)

            cv2.putText(frame, f'ACTION: {action}', (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Collecting Data", frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        np.save(os.path.join(DATA_PATH, action, str(sequence)), keypoints_sequence)

cap.release()
cv2.destroyAllWindows()