import cv2
import mediapipe as mp
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import SEQUENCE_LENGTH, extract_keypoints, normalize_sequence

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def run_extraction(source, label, out_dir, sequence_length=SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(0 if source == "webcam" else source, cv2.CAP_MSMF)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    print("Camera opened successfully. Press q to stop.")

    label_dir = Path(out_dir) / label
    label_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(label_dir.glob("*.npy")))

    sequence = []

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

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            cv2.putText(image, f"Label: {label} | Frame: {len(sequence)}/{sequence_length} | Saved: {existing}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Sign Language Data Collection", image)

            if len(sequence) == sequence_length:
                normalized = normalize_sequence(sequence)
                save_path = label_dir / f"{label}_{existing:04d}.npy"
                np.save(save_path, normalized)
                print(f"Saved sequence: {save_path}")
                existing += 1
                sequence = []

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_extraction(source="webcam", label="sorry", out_dir=PROJECT_ROOT / "data" / "raw")
