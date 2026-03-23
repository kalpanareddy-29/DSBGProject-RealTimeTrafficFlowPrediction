# ==========================================
# IMPORTS
# ==========================================
from kafka import KafkaConsumer
import json
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import os

# ==========================================
# PATH SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
ROOT_DIR = os.path.dirname(BASE_DIR)                    # project root

MODEL_PATH = os.path.join(ROOT_DIR, "models", "traffic_bilstm_attention.keras")
SCALER_PATH = os.path.join(ROOT_DIR, "models", "scaler.save")
OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "stream_predictions.csv")

# ==========================================
# LOAD MODEL + SCALER
# ==========================================
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ==========================================
# KAFKA CONSUMER
# ==========================================
consumer = KafkaConsumer(
    "traffic-data",
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("🧠 Consumer started...\n")

# ==========================================
# BUFFER FOR LAST 12 VALUES
# ==========================================
buffer = []
count = 0

# ==========================================
# MAIN LOOP
# ==========================================
for msg in consumer:

    data = msg.value

    timestamp = data.get("timestamp", "unknown")
    values = list(data.values())[1:]

    buffer.append(values)

    if len(buffer) > 12:
        buffer.pop(0)

    if len(buffer) == 12:

        X = np.array(buffer)
        X_scaled = scaler.transform(X)
        X_scaled = np.expand_dims(X_scaled, axis=0)

        pred = model.predict(X_scaled)
        pred = pred[0][0]

        pred_real = scaler.inverse_transform(pred.reshape(1, -1))[0]

        actual = np.array(values)
        error = np.abs(pred_real - actual)

        count += 1

        print(f"\n🕒 Timestamp: {timestamp}")
        print(f"Processed Row: {count}")

        for i in range(5):
            print(
                f"Sensor {i} | "
                f"Actual: {actual[i]:.2f} | "
                f"Predicted: {pred_real[i]:.2f} | "
                f"Error: {error[i]:.2f}"
            )

        # ==========================================
        # SAVE SENSOR-WISE DATA
        # ==========================================
        records = []

        for i in range(len(pred_real)):
            records.append({
                "timestamp": timestamp,
                "sensor_id": i,
                "actual": round(actual[i], 2),
                "predicted": round(pred_real[i], 2),
                "error": round(error[i], 2)
            })

        df = pd.DataFrame(records)

        df.to_csv(
            OUTPUT_PATH,
            mode='a',
            header=not os.path.exists(OUTPUT_PATH),
            index=False
        )