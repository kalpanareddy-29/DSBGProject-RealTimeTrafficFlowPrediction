# ==========================================
# IMPORTS
# ==========================================
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(layout="wide", page_title="🚦 Traffic Dashboard")
st.title("🚦 Real-Time Traffic Dashboard")

st.success("✅ Live data streaming active")

# ==========================================
# AUTO REFRESH (30 sec)
# ==========================================
st.markdown(
    """
    <script>
        setTimeout(function(){
            window.location.reload();
        }, 30000);
    </script>
    """,
    unsafe_allow_html=True
)

# ==========================================
# PATH SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "data", "stream_predictions.csv")
LOC_PATH = os.path.join(ROOT_DIR, "data", "locations.csv")

# ==========================================
# LOAD DATA
# ==========================================
if not os.path.exists(DATA_PATH):
    st.error("❌ stream_predictions.csv not found")
    st.stop()

df = pd.read_csv(DATA_PATH)

if df.empty:
    st.warning("⏳ No data yet. Run producer + consumer.")
    st.stop()

# ==========================================
# CLEAN DATA
# ==========================================
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["sensor_id"] = pd.to_numeric(df["sensor_id"], errors="coerce")

df = df.dropna(subset=["timestamp", "sensor_id"])
df["sensor_id"] = df["sensor_id"].astype(int)

df = df.sort_values("timestamp")

# ==========================================
# LOAD LOCATION DATA
# ==========================================
if not os.path.exists(LOC_PATH):
    st.error("❌ locations.csv not found")
    st.stop()

loc_df = pd.read_csv(LOC_PATH)
loc_df.columns = [c.lower() for c in loc_df.columns]

# ==========================================
# LATEST DATA
# ==========================================
latest = df.groupby("sensor_id").tail(1)
latest_timestamp = latest["timestamp"].max()

# ==========================================
# METRICS
# ==========================================
mae = np.mean(np.abs(latest["actual"] - latest["predicted"]))
rmse = np.sqrt(np.mean((latest["actual"] - latest["predicted"]) ** 2))
mape = np.mean(
    np.abs((latest["actual"] - latest["predicted"]) /
           (latest["actual"].replace(0, np.nan) + 1e-5))
) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("📏 MAE", f"{mae:.2f}")
c2.metric("📐 RMSE", f"{rmse:.2f}")
c3.metric("📊 MAPE", f"{mape:.2f}%")
c4.metric("🕒 Last Update", str(latest_timestamp)[:19])

st.divider()

# ==========================================
# SENSOR SELECTOR
# ==========================================
max_sensor = int(df["sensor_id"].max())
sensor_id = st.slider("🔍 Select Sensor ID", 0, max_sensor, 0)

# ==========================================
# CURRENT STATUS
# ==========================================
st.subheader("🚗 Current Status")

def classify(speed):
    if speed < 40:
        return "🔴 Heavy"
    elif speed < 80:
        return "🟡 Medium"
    else:
        return "🟢 Free"

current = latest[latest["sensor_id"] == sensor_id]

if not current.empty:
    row = current.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Actual", f"{row['actual']:.2f}")
    c2.metric("Predicted", f"{row['predicted']:.2f}")
    c3.metric("Traffic", classify(row["actual"]))
else:
    st.info("No current data")

# ==========================================
# LIVE GAUGE
# ==========================================
st.subheader("🚦 Live Traffic Gauge")

if not current.empty:
    speed = row["actual"]

    g1, g2 = st.columns(2)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=speed,
        title={"text": "Speed"},
        gauge={
            "axis": {"range": [0, 120]},
            "bar": {"color": "blue"},
            "steps": [
                {"range": [0, 40], "color": "red"},
                {"range": [40, 80], "color": "orange"},
                {"range": [80, 120], "color": "green"},
            ],
        },
    ))

    g1.plotly_chart(fig_gauge, width="stretch")

    def congestion_label(speed):
        if speed < 40:
            return "🔴 Heavy Traffic"
        elif speed < 80:
            return "🟡 Moderate Traffic"
        else:
            return "🟢 Free Flow"

    g2.metric("Congestion Level", congestion_label(speed))

st.divider()

# ==========================================
# LAST 2 HOURS CHART
# ==========================================
latest_time = df["timestamp"].max()
two_hours_ago = latest_time - pd.Timedelta(hours=2)

sensor_df = df[
    (df["sensor_id"] == sensor_id) &
    (df["timestamp"] >= two_hours_ago)
]

sensor_df = sensor_df.sort_values("timestamp")

st.subheader(f"📈 Sensor {sensor_id} — Last 2 Hours")

if not sensor_df.empty:
    fig = px.line(sensor_df, x="timestamp", y=["actual", "predicted"])
    st.plotly_chart(fig, width="stretch")
else:
    st.warning("No data for this sensor")

# ==========================================
# LAST 60 MIN CHART (NEW)
# ==========================================
st.subheader("⏱️ Last 60 Minutes Trend")

one_hour_ago = latest_time - pd.Timedelta(minutes=60)

last_60_df = df[
    (df["sensor_id"] == sensor_id) &
    (df["timestamp"] >= one_hour_ago)
]

if not last_60_df.empty:
    fig_60 = px.line(last_60_df, x="timestamp", y=["actual", "predicted"])
    st.plotly_chart(fig_60, width="stretch")
else:
    st.warning("No data for last 60 minutes")

st.divider()

# ==========================================
# LAST 4 READINGS
# ==========================================
st.subheader("📊 Recent Predictions (Last 4)")

recent_df = df[df["sensor_id"] == sensor_id].sort_values("timestamp").tail(4)

if not recent_df.empty:

    mae_recent = np.mean(np.abs(recent_df["actual"] - recent_df["predicted"]))
    mse_recent = np.mean((recent_df["actual"] - recent_df["predicted"]) ** 2)

    m1, m2 = st.columns(2)
    m1.metric("📏 MAE (Last 4)", f"{mae_recent:.2f}")
    m2.metric("📐 MSE (Last 4)", f"{mse_recent:.2f}")

    st.dataframe(
        recent_df[["timestamp", "actual", "predicted", "error"]],
        width="stretch"
    )

else:
    st.warning("No recent data")

st.divider()

# ==========================================
# MAP
# ==========================================
st.subheader("🗺️ Traffic Map")

def safe_lookup(sensor_id, col):
    try:
        return loc_df.iloc[int(sensor_id)][col]
    except:
        return None

latest = latest.copy()
latest["lat"] = latest["sensor_id"].apply(lambda x: safe_lookup(x, "latitude"))
latest["lon"] = latest["sensor_id"].apply(lambda x: safe_lookup(x, "longitude"))
latest["name"] = latest["sensor_id"].apply(lambda x: safe_lookup(x, "name"))

map_data = latest.dropna(subset=["lat", "lon"]).copy()

map_data["congestion"] = map_data["actual"].apply(
    lambda x: "Heavy" if x < 40 else ("Medium" if x < 80 else "Free")
)

if not map_data.empty:
    fig_map = px.scatter_map(
        map_data,
        lat="lat",
        lon="lon",
        color="congestion",
        size=np.clip(map_data["actual"], 10, 100),
        hover_name="name",
        zoom=9,
        height=600
    )
    st.plotly_chart(fig_map, width="stretch")
else:
    st.warning("No map data available")