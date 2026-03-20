import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

model = load_model("action_model.h5")

actions = np.array(['eating','drinking','walking','sitting','waving'])

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
    running_mode=VisionRunningMode.VIDEO
)

landmarker = PoseLandmarker.create_from_options(options)

sequence = []

cap = cv2.VideoCapture(0)
timestamp = 0

while cap.isOpened():

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

        sequence.append(keypoints)

        if len(sequence) > 30:
            sequence = sequence[-30:]

        if len(sequence) == 30:

            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            action = actions[np.argmax(res)]

            cv2.putText(frame, action, (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

    cv2.imshow("Action Recognition", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()