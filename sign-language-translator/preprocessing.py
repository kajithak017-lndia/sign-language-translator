from __future__ import annotations

import numpy as np

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 258


def extract_keypoints(results) -> np.ndarray:
    pose = (
        np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4, dtype=np.float32)
    )
    left_hand = (
        np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3, dtype=np.float32)
    )
    right_hand = (
        np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3, dtype=np.float32)
    )
    return np.concatenate([pose, left_hand, right_hand]).astype(np.float32)


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.shape != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(
            f"Expected sequence shape {(SEQUENCE_LENGTH, FEATURE_COUNT)}, got {sequence.shape}"
        )

    normalized = sequence.copy()
    for frame_index in range(normalized.shape[0]):
        frame = normalized[frame_index]
        pose = frame[:132].reshape(33, 4).copy()
        left_hand = frame[132:195].reshape(21, 3).copy()
        right_hand = frame[195:258].reshape(21, 3).copy()

        center = pose[0, :3].copy()
        scale = float(np.linalg.norm(pose[11, :3] - pose[12, :3]))
        if scale < 1e-6:
            scale = 1.0

        pose[:, :3] = (pose[:, :3] - center) / scale
        left_hand = (left_hand - center) / scale
        right_hand = (right_hand - center) / scale
        normalized[frame_index] = np.concatenate([pose.flatten(), left_hand.flatten(), right_hand.flatten()])

    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_batch(sequences: np.ndarray) -> np.ndarray:
    return np.stack([normalize_sequence(sequence) for sequence in sequences]).astype(np.float32)

