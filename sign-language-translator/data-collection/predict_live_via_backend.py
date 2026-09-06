import cv2
import mediapipe as mp
import numpy as np
import requests
import sys
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import SEQUENCE_LENGTH, extract_keypoints

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

BACKEND_URL = "http://127.0.0.1:8080/api/predict-sign"

sequence = deque(maxlen=SEQUENCE_LENGTH)
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
        if len(sequence) == SEQUENCE_LENGTH:
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
