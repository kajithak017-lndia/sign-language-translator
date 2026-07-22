import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

model = load_model("models/sign_model.h5")
with open("models/sign_model.labels.txt") as f:
    labels = f.read().splitlines()

print("Loaded model. Signs it knows:", labels)


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
            input_data = np.expand_dims(np.array(sequence), axis=0)
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