"""
Phase 1 - Step 3: Flask API
=============================
Exposes two endpoints:
  POST /predict  →  { "priority": "High", "confidence": 0.87 }
  GET  /health   →  { "status": "ok" }
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ─────────────────────────────────────────
# Load the trained model at startup
# ─────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully.")


# ─────────────────────────────────────────
# Health Check Endpoint
# ─────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Returns a simple health status — used by K6 tests."""
    return jsonify({"status": "ok"}), 200


# ─────────────────────────────────────────
# Predict Endpoint
# ─────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: { "Task": "Fix production bug" }
    Returns JSON: { "Priority": "High", "confidence": 0.87 }
    """
    data = request.get_json()

    # Input validation
    if not data or "task" not in data:
        return jsonify({"error": "Missing 'task' field in request body"}), 400

    task_text = data["task"].strip()
    if not task_text:
        return jsonify({"error": "'task' field cannot be empty"}), 400

    # Run prediction
    prediction = model.predict([task_text])[0]
    probabilities = model.predict_proba([task_text])[0]
    confidence = float(np.max(probabilities))

    return jsonify({
        "priority": prediction,
        "confidence": round(confidence, 2)
    }), 200


# ─────────────────────────────────────────
# Run the app
# ─────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
