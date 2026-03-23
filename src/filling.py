import pandas as pd
import numpy as np

# ======================
# Load dataset
# ======================
df = pd.read_csv("C:\\Users\\pamal\\OneDrive\\Desktop\\Assignment\\datas\\data\\data.csv")

# ✅ Rename 'index' → 'timestamp'
df.rename(columns={"index": "timestamp"}, inplace=True)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by time
df = df.sort_values("timestamp")

# ======================
# Separate sensor columns
# ======================
sensor_cols = df.columns.drop("timestamp")

# ======================
# Count zero values
# ======================
total_values = df[sensor_cols].size
zero_values = (df[sensor_cols] == 0.0).sum().sum()

print("Total values:", total_values)
print("Zero values:", zero_values)
print("Percentage of zeros:", (zero_values / total_values) * 100)

# ======================
# Replace zeros with NaN
# ======================
df[sensor_cols] = df[sensor_cols].replace(0, np.nan)

# ======================
# Interpolation (time-based)
# ======================
df[sensor_cols] = df[sensor_cols].interpolate(method="linear", axis=0)

# ======================
# Fill remaining edges
# ======================
df[sensor_cols] = df[sensor_cols].bfill().ffill()

# ======================
# Round values
# ======================
df[sensor_cols] = df[sensor_cols].round(2)

# ======================
# Verify missing values
# ======================
remaining_missing = df[sensor_cols].isna().sum().sum()
print("Remaining missing values after filling:", remaining_missing)

# ======================
# Save cleaned dataset
# ======================
df.to_csv("C:\\Users\\pamal\\OneDrive\\Desktop\\Assignment\\datas\\data\\data.csv", index=False)

print("✅ Cleaned dataset saved successfully")

df = pd.read_csv("C:\\Users\\pamal\\OneDrive\\Desktop\\Assignment\\datas\\data\\location.csv")

# ======================
# Total null values
# ======================
total_nulls = df.isnull().sum().sum()
print("Total missing values:", total_nulls)

# ======================
# Column-wise nulls
# ======================
print("\nMissing values per column:")
print(df.isnull().sum())