from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import FEATURE_COUNT, SEQUENCE_LENGTH, normalize_sequence


if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data" / "raw"
    count = 0
    for label_dir in data_dir.iterdir():
        if not label_dir.is_dir():
            continue
        for file in label_dir.glob("*.npy"):
            seq = np.load(file)
            if seq.shape != (SEQUENCE_LENGTH, FEATURE_COUNT):
                print(f"Skipping {file}: expected {(SEQUENCE_LENGTH, FEATURE_COUNT)}, got {seq.shape}")
                continue
            normalized = normalize_sequence(seq)
            np.save(file, normalized)
            count += 1
    print(f"Normalized {count} sequences in place.")
