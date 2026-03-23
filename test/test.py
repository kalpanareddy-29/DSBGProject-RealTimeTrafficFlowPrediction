# ==========================================
# IMPORTS
# ==========================================
import pandas as pd
import numpy as np
import joblib
import requests
import os
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# ==========================================
# PATH SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_bilstm_attention.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.save")
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
LOC_PATH = os.path.join(BASE_DIR, "data", "locations.csv")

# ==========================================
# LOAD ENV
# ==========================================
load_dotenv()
API_KEY = os.getenv("TOMTOM_API_KEY")

if not API_KEY:
    print("⚠️ WARNING: TOMTOM_API_KEY not found")

# ==========================================
# LOAD MODEL
# ==========================================
print("📦 Loading model...")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("✅ Model loaded")

# ==========================================
# LOAD DATA
# ==========================================
print("📊 Loading dataset...")

df = pd.read_csv(DATA_PATH)

df.rename(columns={"index": "timestamp"}, inplace=True, errors="ignore")

sensor_cols = df.columns[1:]
data = df[sensor_cols].values

data_scaled = scaler.transform(data)

# ==========================================
# PREDICTION (LAST 12 VALUES)
# ==========================================
window = 12

last_seq = data_scaled[-window:]
last_seq = np.expand_dims(last_seq, axis=0)

pred = model.predict(last_seq)
pred = pred[0][0]

pred_real = scaler.inverse_transform(pred.reshape(1, -1))[0]

print("✅ Prediction ready")

# ==========================================
# LOAD LOCATION
# ==========================================
print("📍 Loading locations...")

loc_df = pd.read_csv(LOC_PATH)
loc_df.columns = [c.lower() for c in loc_df.columns]

# ==========================================
# FILTER URBAN-LIKE SENSORS
# ==========================================
urban_df = loc_df[
    (~loc_df["name"].str.contains("FWY", case=False, na=False)) &
    (~loc_df["name"].str.contains("HWY", case=False, na=False))
]

if len(urban_df) < 10:
    urban_df = loc_df.sample(min(15, len(loc_df)))

# ==========================================
# TOMTOM API FUNCTION
# ==========================================
def get_traffic(lat, lon):

    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/12/json"

    params = {
        "point": f"{lat},{lon}",
        "key": API_KEY
    }

    try:
        res = requests.get(url, params=params, timeout=5)
        data = res.json()

        return data["flowSegmentData"]["currentSpeed"]

    except Exception as e:
        print("API error:", e)
        return None

# ==========================================
# CONGESTION CLASS
# ==========================================
def classify(speed):
    if speed < 40:
        return "Heavy"
    elif speed < 80:
        return "Medium"
    else:
        return "Free"

# ==========================================
# TERMINAL OUTPUT
# ==========================================
print("\n📊 REAL-TIME COMPARISON (URBAN FILTERED)\n")

count = 0

for idx, row in urban_df.iterrows():

    if count >= 15:
        break

    lat = row["latitude"]
    lon = row["longitude"]
    name = row.get("name", "Unknown")
    fwy = row.get("fwy", "N/A")

    # safe index mapping
    if idx >= len(pred_real):
        continue

    pred_speed = pred_real[idx]

    real_speed = get_traffic(lat, lon)

    if real_speed is None:
        continue

    # filter unrealistic speeds
    if real_speed > 120:
        continue

    error = abs(pred_speed - real_speed)

    if real_speed != 0:
        accuracy = 100 - (error / real_speed * 100)
    else:
        accuracy = 0

    print(f"🔹 Sensor Index : {idx}")
    print(f"Location       : {name}")
    print(f"Highway        : {fwy}")
    print(f"Coordinates    : ({lat:.4f}, {lon:.4f})")

    print(f"Predicted Speed: {pred_speed:.2f} km/h")
    print(f"Real Speed     : {real_speed:.2f} km/h")

    print(f"Traffic        : {classify(pred_speed)} vs {classify(real_speed)}")

    print(f"Difference     : {error:.2f}")
    print(f"Accuracy       : {accuracy:.2f}%")

    print("-" * 60)

    count += 1