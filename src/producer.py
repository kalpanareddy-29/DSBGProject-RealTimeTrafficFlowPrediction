from kafka import KafkaProducer
import pandas as pd
import json
import time
import os

# ==========================================
# PATH SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
ROOT_DIR = os.path.dirname(BASE_DIR)                    # project root

DATA_PATH = os.path.join(ROOT_DIR, "data", "data.csv")

# ==========================================
# KAFKA PRODUCER
# ==========================================
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv(DATA_PATH)

print("🚀 Starting Producer...\n")

# ==========================================
# STREAM DATA
# ==========================================
for i in range(len(df)):

    row = df.iloc[i].to_dict()

    producer.send("traffic-data", value=row)

    print(f"Sent row {i}")

    time.sleep(0.01)   