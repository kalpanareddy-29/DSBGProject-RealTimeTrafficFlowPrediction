# ==========================================
# IMPORTS
# ==========================================
import os
import sys
import pandas as pd
import numpy as np
import joblib
import requests
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import plotly.express as px

# ==========================================
# PATHS — works from any folder
# ==========================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data",   "data.csv")
LOC_PATH    = os.path.join(BASE_DIR, "data",   "location.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "models", "traffic_bilstm_attention.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.save")

# ==========================================
# LOAD ENV (API KEY)
# ==========================================
load_dotenv(os.path.join(BASE_DIR, ".env"))
API_KEY = os.getenv("TOMTOM_API_KEY")

if not API_KEY:
    print("ERROR: TOMTOM_API_KEY not found in .env")
    sys.exit(1)

# ==========================================
# LOAD MODEL + SCALER
# ==========================================
print("Loading model...")
model  = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Model loaded")

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv(DATA_PATH)
df.rename(columns={"index": "timestamp"}, inplace=True, errors="ignore")
df["timestamp"] = pd.to_datetime(df["timestamp"])

sensor_cols = [c for c in df.columns if c != "timestamp"]
data        = df[sensor_cols].values
data_scaled = scaler.transform(data)

# ==========================================
# TAKE LAST 12 ROWS AS INPUT WINDOW
# ==========================================
window  = 12
X_input = np.expand_dims(data_scaled[-window:], axis=0)  # (1, 12, n_sensors)
y_true  = data[-1]                                        # last actual row

# ==========================================
# PREDICT (15-min forecast)
# ==========================================
pred      = model.predict(X_input, verbose=0)
pred_step = pred[0][0]                                    # (n_sensors,)
pred_real = scaler.inverse_transform(
    pred_step.reshape(1, -1)
)[0]                                                      # mph

print("Prediction ready")

# ==========================================
# LOAD LOCATION DATA
# ==========================================
loc_df = pd.read_csv(LOC_PATH)
loc_df.columns = [c.lower() for c in loc_df.columns]

# ==========================================
# TOMTOM API — FIX: convert km/h to mph
# TomTom always returns km/h
# PEMS-BAY model predicts in mph
# Must convert before comparing
# ==========================================
def get_traffic_mph(lat, lon):
    url = (
        f"https://api.tomtom.com/traffic/services/4/flowSegmentData/"
        f"absolute/12/json?point={lat},{lon}&key={API_KEY}"
    )
    try:
        res       = requests.get(url, timeout=5).json()
        speed_kmh = res["flowSegmentData"]["currentSpeed"]
        speed_mph = round(speed_kmh * 0.621371, 2)        # km/h to mph
        return speed_mph
    except Exception:
        return None

# ==========================================
# CLASSIFICATION — FIX: thresholds in mph
# (old code used 40/80 which are km/h values)
# ==========================================
def classify(speed_mph):
    if speed_mph is None:
        return "Unknown"
    if speed_mph < 25:
        return "Heavy"
    elif speed_mph < 50:
        return "Medium"
    else:
        return "Free"

# ==========================================
# PROCESS 3 CORRIDORS
# ==========================================
corridor_1 = loc_df.iloc[0:10]
corridor_2 = loc_df.iloc[50:60]
corridor_3 = loc_df.iloc[100:110]

records = []

def process_corridor(corridor, name):
    print(f"\n{name}")
    print("-" * 50)

    for idx, row in corridor.iterrows():
        if idx >= len(pred_real):
            continue

        lat       = row["latitude"]
        lon       = row["longitude"]
        location  = row["name"]
        predicted = round(float(pred_real[idx]), 2)
        actual    = round(float(y_true[idx]),    2)

        # FIX: use get_traffic_mph (converts km/h to mph)
        api_speed = get_traffic_mph(lat, lon)

        print(f"  {location}")
        print(f"    Predicted (mph) : {predicted}")
        print(f"    Actual    (mph) : {actual}")
        print(f"    TomTom    (mph) : {api_speed}")

        if api_speed is None:
            continue

        error = round(abs(predicted - actual), 2)

        records.append({
            "lat":        lat,
            "lon":        lon,
            "corridor":   name,
            "location":   location,
            "predicted":  predicted,
            "actual":     actual,
            "api_mph":    api_speed,
            "error":      error,
            "congestion": classify(api_speed),
        })

# ==========================================
# RUN ALL CORRIDORS
# ==========================================
process_corridor(corridor_1, "Corridor 1 — Hwy 17 N")
process_corridor(corridor_2, "Corridor 2 — Hwy 85 N")
process_corridor(corridor_3, "Corridor 3 — Hwy 101 N")

# ==========================================
# BUILD DATAFRAME
# ==========================================
map_df = pd.DataFrame(records)
print(f"\nTotal points: {len(map_df)}")

if map_df.empty:
    print("No data — check your TomTom API key")
    sys.exit(1)

# ==========================================
# PLOTLY MAP
# ==========================================
fig = px.scatter_mapbox(
    map_df,
    lat="lat",
    lon="lon",
    color="congestion",
    color_discrete_map={"Heavy": "red", "Medium": "orange", "Free": "green", "Unknown": "gray"},
    size="error",
    size_max=20,
    hover_name="location",
    hover_data={
        "corridor":  True,
        "predicted": True,
        "actual":    True,
        "api_mph":   True,
        "error":     True,
        "lat":       False,
        "lon":       False,
    },
    mapbox_style="open-street-map",
    zoom=9,
    center={"lat": loc_df["latitude"].mean(), "lon": loc_df["longitude"].mean()},
    height=700,
    title="Traffic Prediction vs TomTom API — 3 Corridors"
)

# Save HTML + show
out_path = os.path.join(BASE_DIR, "urban_traffic_map.html")
fig.write_html(out_path)
print(f"Map saved to urban_traffic_map.html")

fig.show()
