import cv2
import mediapipe as mp
import numpy as np
import sys
from collections import deque
from pathlib import Path
from tensorflow.keras.models import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import SEQUENCE_LENGTH, extract_keypoints, normalize_sequence

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

model = load_model(PROJECT_ROOT / "models" / "sign_model.h5")
with open(PROJECT_ROOT / "models" / "sign_model.labels.txt") as f:
    labels = f.read().splitlines()

print("Loaded model. Signs it knows:", labels)


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
            input_data = np.expand_dims(normalize_sequence(np.array(sequence)), axis=0)
            prediction = model.predict(input_data, verbose=0)[0]
            best_idx = np.argmax(prediction)
            confidence = prediction[best_idx]
            prediction_text = f"{labels[best_idx]} ({confidence*100:.0f}%)"

        cv2.putText(image, prediction_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Sign Prediction", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
