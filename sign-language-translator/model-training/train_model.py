import numpy as np
from pathlib import Path
import sys
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from preprocessing import FEATURE_COUNT, SEQUENCE_LENGTH, normalize_batch


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    labels = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    label_map = {label: idx for idx, label in enumerate(labels)}

    X, y = [], []
    for label in labels:
        for file in (data_dir / label).glob("*.npy"):
            sequence = np.load(file)
            if sequence.shape != (SEQUENCE_LENGTH, FEATURE_COUNT):
                print(f"Skipping {file}: expected {(SEQUENCE_LENGTH, FEATURE_COUNT)}, got {sequence.shape}")
                continue
            X.append(sequence)
            y.append(label_map[label])

    if not X:
        raise RuntimeError(f"No valid .npy sequences found in {data_dir}")

    return normalize_batch(np.array(X)), np.array(y), labels


def augment_sequence(sequence):
    augmented = sequence.copy()
    coordinate_mask = np.ones(FEATURE_COUNT, dtype=bool)
    coordinate_mask[3:132:4] = False
    augmented[:, coordinate_mask] += np.random.normal(0, 0.015, size=augmented[:, coordinate_mask].shape)
    shift = np.random.randint(-2, 3)
    if shift > 0:
        augmented[shift:] = sequence[:-shift]
        augmented[:shift] = sequence[0]
    elif shift < 0:
        offset = abs(shift)
        augmented[:-offset] = sequence[offset:]
        augmented[-offset:] = sequence[-1]
    return augmented.astype(np.float32)


def make_augmented_training_data(X_train, y_train, copies=2):
    X_augmented = [X_train]
    y_augmented = [y_train]
    for _ in range(copies):
        X_augmented.append(np.stack([augment_sequence(sequence) for sequence in X_train]))
        y_augmented.append(y_train)
    return np.concatenate(X_augmented), np.concatenate(y_augmented)

def build_model(seq_len, num_features, num_classes):
    inputs = Input(shape=(seq_len, num_features))

    x = Conv1D(64, kernel_size=5, activation="relu", padding="same", kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=3, activation="relu", padding="same", kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Bidirectional(LSTM(96, return_sequences=True, dropout=0.2))(x)
    x = Bidirectional(LSTM(48, return_sequences=True, dropout=0.2))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(96, activation="relu", kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.35)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data" / "raw"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y, labels = load_dataset(data_dir)
    print(f"Loaded {len(X)} examples across {len(labels)} signs: {labels}")
    print("Class counts:", dict(Counter(y)))

    y_cat = to_categorical(y, num_classes=len(labels))
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )
    X_train, y_train = make_augmented_training_data(X_train, y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(labels)),
        y=np.argmax(y_train, axis=1),
    )
    class_weight = {idx: float(weight) for idx, weight in enumerate(class_weights)}

    model = build_model(seq_len=X.shape[1], num_features=X.shape[2], num_classes=len(labels))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
        ModelCheckpoint(models_dir / "sign_model.h5", monitor="val_accuracy", save_best_only=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=180,
        batch_size=16,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    with open(models_dir / "sign_model.labels.txt", "w") as f:
        f.write("\n".join(labels))

    print(f"Training done! Model saved to {models_dir / 'sign_model.h5'}")
