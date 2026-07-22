import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    labels = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    label_map = {label: idx for idx, label in enumerate(labels)}

    X, y = [], []
    for label in labels:
        for file in (data_dir / label).glob("*.npy"):
            X.append(np.load(file))
            y.append(label_map[label])

    return np.array(X), np.array(y), labels

def build_model(seq_len, num_features, num_classes):
    inputs = Input(shape=(seq_len, num_features))

    x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = Dropout(0.3)(x)

    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    X, y, labels = load_dataset("data/raw")
    print(f"Loaded {len(X)} examples across {len(labels)} signs: {labels}")

    y_cat = to_categorical(y, num_classes=len(labels))
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(seq_len=X.shape[1], num_features=X.shape[2], num_classes=len(labels))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ModelCheckpoint("models/sign_model.h5", monitor="val_accuracy", save_best_only=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=8,
        callbacks=callbacks,
    )

    with open("models/sign_model.labels.txt", "w") as f:
        f.write("\n".join(labels))

    print("Training done! Model saved to models/sign_model.h5")