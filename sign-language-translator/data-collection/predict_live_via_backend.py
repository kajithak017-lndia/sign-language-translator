import cv2
import mediapipe as mp
import numpy as np
import requests
from collections import deque

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

BACKEND_URL = "http://127.0.0.1:8080/api/predict-sign"


def extract_keypoints(results):
    lh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(21 * 3)
    )
    rh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(21 * 3)
    )
    pose = (
        np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(33 * 4)
    )
    return np.concatenate([pose, lh, rh])


sequence = deque(maxlen=30)
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        sequence.append(extract_keypoints(results))

        prediction_text = "..."
        if len(sequence) == 30:
            payload = {"sequence": np.array(sequence).tolist()}
            try:
                response = requests.post(BACKEND_URL, json=payload, timeout=5)
                result = response.json()
                prediction_text = f"{result['sign']} ({result['confidence']*100:.0f}%)"
            except Exception as e:
                prediction_text = f"Error: {e}"

        cv2.putText(image, prediction_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Prediction via Java Backend", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()