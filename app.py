# ==========================================
# IMPORTS
# ==========================================
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import requests
import os
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from datetime import datetime
import random

# ==========================================
# INIT APP
# ==========================================
app = Flask(__name__)

print("🚀 Starting Traffic API...")

# ==========================================
# PATH SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_bilstm_attention.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.save")
LOC_PATH = os.path.join(BASE_DIR, "data", "locations.csv")

# ==========================================
# LOAD ENV (TOMTOM API KEY)
# ==========================================
load_dotenv()
API_KEY = os.getenv("TOMTOM_API_KEY")

# ==========================================
# LOAD MODEL
# ==========================================
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model Loaded")
except Exception as e:
    print("❌ Model error:", e)
    model, scaler = None, None

# ==========================================
# LOAD LOCATION DATA
# ==========================================
try:
    loc_df = pd.read_csv(LOC_PATH)
    loc_df.columns = [c.lower() for c in loc_df.columns]
    print("✅ Location data loaded")
except Exception as e:
    print("❌ Location error:", e)
    loc_df = None

# ==========================================
# HELPER FUNCTION
# ==========================================
def classify(speed):
    if speed < 40:
        return "🔴 Heavy"
    elif speed < 80:
        return "🟡 Medium"
    else:
        return "🟢 Free"

# ==========================================
# HOME PAGE
# ==========================================
@app.route("/")
def home():
    return render_template("index.html")

# ==========================================
# HEALTH CHECK
# ==========================================
@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None
    })

# ==========================================
# LIVE TRAFFIC (TomTom API)
# ==========================================
@app.route("/live-traffic")
def live_traffic():

    if loc_df is None:
        return jsonify({"error": "No location data"})

    if not API_KEY:
        return jsonify({"error": "Missing TOMTOM API KEY"})

    results = []

    sample = loc_df.sample(min(5, len(loc_df)))

    for _, row in sample.iterrows():

        lat = row["latitude"]
        lon = row["longitude"]

        try:
            url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
            params = {
                "point": f"{lat},{lon}",
                "key": API_KEY
            }

            res = requests.get(url, params=params, timeout=5)
            data = res.json()

            speed = data["flowSegmentData"]["currentSpeed"]

            results.append({
                "name": row.get("name", "Unknown"),
                "lat": lat,
                "lon": lon,
                "speed": speed,
                "traffic": classify(speed),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": round(random.uniform(25, 35), 1)
            })

        except Exception as e:
            print("TomTom error:", e)
            continue

    return jsonify(results)

# ==========================================
# PREDICT API
# ==========================================
@app.route("/predict", methods=["POST"])
def predict():

    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json

    if not data or "sequence" not in data:
        return jsonify({"error": "Missing 'sequence'"}), 400

    try:
        seq = np.array(data["sequence"])

        # Ensure correct shape
        if len(seq.shape) == 1:
            seq = seq.reshape(-1, 1)

        # Scale
        seq_scaled = scaler.transform(seq)

        # Model expects 3D input
        seq_scaled = np.expand_dims(seq_scaled, axis=0)

        # Predict
        pred = model.predict(seq_scaled)
        pred = pred[0][0]

        # Inverse transform
        pred_real = scaler.inverse_transform(pred.reshape(1, -1))[0]

        # Take next 4 predictions
        next_4 = pred_real[:4].tolist()

        avg_speed = float(np.mean(next_4))

        return jsonify({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "next_4_predictions": next_4,
            "avg_speed": avg_speed,
            "traffic": classify(avg_speed)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# RUN SERVER
# ==========================================
if __name__ == "__main__":
    app.run(debug=True)