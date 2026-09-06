from flask import Flask, request, jsonify
import numpy as np
from pathlib import Path
import sys
from tensorflow.keras.models import load_model

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import FEATURE_COUNT, SEQUENCE_LENGTH, normalize_sequence

model = load_model(PROJECT_ROOT / "models" / "sign_model.h5")
with open(PROJECT_ROOT / "models" / "sign_model.labels.txt") as f:
    labels = f.read().splitlines()


print("Model loaded. Known signs:", labels)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "sequence" not in data:
        return jsonify({"error": "Missing 'sequence' in request"}), 400

    sequence = np.array(data["sequence"])

    if sequence.shape != (SEQUENCE_LENGTH, FEATURE_COUNT):
        return jsonify({"error": f"Expected shape {(SEQUENCE_LENGTH, FEATURE_COUNT)}, got {sequence.shape}"}), 400

    input_data = np.expand_dims(normalize_sequence(sequence), axis=0)
    prediction = model.predict(input_data, verbose=0)[0]

    best_idx = int(np.argmax(prediction))
    confidence = float(prediction[best_idx])

    return jsonify({
        "sign": labels[best_idx],
        "confidence": confidence,
        "all_scores": {labels[i]: float(prediction[i]) for i in range(len(labels))}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
