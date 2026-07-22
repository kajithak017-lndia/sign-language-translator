from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("models/sign_model.h5")
with open("models/sign_model.labels.txt") as f:
    labels = f.read().splitlines()


print("Model loaded. Known signs:", labels)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "sequence" not in data:
        return jsonify({"error": "Missing 'sequence' in request"}), 400

    sequence = np.array(data["sequence"])

    if sequence.shape != (30, 258):
        return jsonify({"error": f"Expected shape (30, 258), got {sequence.shape}"}), 400

    input_data = np.expand_dims(sequence, axis=0)
    prediction = model.predict(input_data, verbose=0)[0]

    best_idx = int(np.argmax(prediction))
    confidence = float(prediction[best_idx])

    return jsonify({
        "sign": labels[best_idx],
        "confidence": confidence,
        "all_scores": {labels[i]: float(prediction[i]) for i in range(len(labels))}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)