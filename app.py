"""
CardioPredict AI — Flask REST API v2
=====================================
Endpoints:
  GET  /api/health        — liveness check
  GET  /api/model-info    — model metadata + metrics
  POST /api/predict       — single patient prediction
  GET  /api/user/<uid>/history  — prediction history stub

Deploy on Render: gunicorn -w 2 -b 0.0.0.0:8000 app:app
"""

import json, traceback
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS

from predict import run_inference, get_feature_names, get_metrics

MODEL_VER = "2.0.0"

app = Flask(__name__)
# Allow all origins in development — restrict to your Firebase domain in prod
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Pre-load model on startup ─────────────────────────────────────────────────
try:
    _feature_names = get_feature_names()
    _metrics       = get_metrics()
    _model_ready   = True
    acc = _metrics.get("test_metrics", {}).get("accuracy", "?")
    print(f"[CardioPredict] ✓ Model loaded  v{MODEL_VER}  "
          f"{len(_feature_names)} features  acc={acc}")
except Exception as exc:
    _model_ready   = False
    _feature_names = []
    _metrics       = {}
    print(f"[CardioPredict] ✗ Model NOT loaded: {exc}")
    print("[CardioPredict]   → Run  python train_model.py  first")


def _err(msg, status=400):
    return jsonify({"error": msg, "status": status}), status


def _validate(data: dict):
    for f in ["age", "sbp", "totalCholesterol"]:
        if data.get(f) is None:
            return f"Missing required field: '{f}'"
    try:
        if not (1 <= int(data["age"]) <= 110):
            return "age must be 1–110"
        if not (50 <= int(data["sbp"]) <= 250):
            return "sbp must be 50–250"
        if not (50 <= int(data["totalCholesterol"]) <= 600):
            return "totalCholesterol must be 50–600"
    except (ValueError, TypeError) as e:
        return f"Invalid numeric value: {e}"
    return None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok" if _model_ready else "degraded",
        "model_ready":   _model_ready,
        "model_version": MODEL_VER,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/model-info", methods=["GET"])
def model_info():
    if not _model_ready:
        return _err("Model not loaded. Run train_model.py first.", 503)
    test_m = _metrics.get("test_metrics", {})
    cv_m   = _metrics.get("cv_summary", {})
    return jsonify({
        "model_version":    MODEL_VER,
        "model_type":       "StackingClassifier (XGB+LGB+RF+ET+SVC → LR)",
        "n_features":       len(_feature_names),
        "feature_names":    _feature_names,
        "test_metrics":     test_m,
        "cv_summary":       cv_m,
        "training_dataset": "UCI Heart Disease (Cleveland+Hungarian+Switzerland+VA)",
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if not _model_ready:
        return _err("Model not loaded. Run train_model.py first.", 503)

    data = request.get_json(silent=True)
    if not data:
        return _err("Request body must be valid JSON with Content-Type: application/json")

    err = _validate(data)
    if err:
        return _err(err, 422)

    try:
        result = run_inference(data)
        return jsonify(result), 200
    except Exception as exc:
        traceback.print_exc()
        return _err(f"Inference error: {exc}", 500)


@app.route("/api/predictions/<string:pred_id>", methods=["GET"])
def get_prediction(pred_id):
    return _err("Prediction history requires a DB backend (stub).", 501)


@app.route("/api/user/<string:uid>/history", methods=["GET"])
def user_history(uid):
    return jsonify([])   # empty until DB is wired up


@app.errorhandler(404)
def not_found(_):    return _err("Endpoint not found", 404)

@app.errorhandler(405)
def method_not_allowed(_): return _err("Method not allowed", 405)


if __name__ == "__main__":
    print(f"\n[CardioPredict] Starting on http://0.0.0.0:8000\n")
    app.run(host="0.0.0.0", port=8000, debug=False)
