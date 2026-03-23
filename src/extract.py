import pandas as pd
import os

# ==============================
# CONFIG
# ==============================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data")

TRAFFIC_FILE = os.path.join(DATA_DIR, "pems-bay.h5")
META_FILE = os.path.join(DATA_DIR, "pems-bay-meta.h5")

# ==============================
# STEP 1: LOAD TRAFFIC DATA
# ==============================
def process_traffic():
    print("📊 Loading traffic data...")

    if not os.path.exists(TRAFFIC_FILE):
        print("❌ Traffic file not found:", TRAFFIC_FILE)
        return

    df = pd.read_hdf(TRAFFIC_FILE)

    print(f"Original Shape: {df.shape}")
    print(df.head())

    # Keep ORIGINAL FORMAT
    df = df.reset_index()

    output_path = os.path.join(OUTPUT_DIR, "data.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Traffic data saved → {output_path}")


# ==============================
# STEP 2: LOAD METADATA
# ==============================
def process_metadata():
    print("📍 Loading metadata...")

    if not os.path.exists(META_FILE):
        print("❌ Metadata file not found:", META_FILE)
        return

    meta = pd.read_hdf(META_FILE)

    print(meta.head())

    output_path = os.path.join(OUTPUT_DIR, "locations.csv")
    meta.to_csv(output_path, index=False)

    print(f"✅ Metadata saved → {output_path}")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("🚀 Starting preprocessing...")

    process_traffic()
    process_metadata()

    print("🔥 DONE — Files ready!")