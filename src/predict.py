# ==========================================
# predict.py
# Evaluates trained model on test set
# Reports RMSE, MAE, MAPE (required by assignment)
# ==========================================

import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# LOAD MODEL + SCALER
# ==========================================
print("Loading model...")

model = load_model("../models/traffic_bilstm_attention.keras")
scaler = joblib.load("../models/scaler.save")

print("Model & scaler loaded")

# ==========================================
# LOAD TEST DATA
# ==========================================
X_test = np.load("../models/X_test.npy")
y_test = np.load("../models/y_test.npy")

print("Test data loaded")

# ==========================================
# PREDICT
# ==========================================
pred = model.predict(X_test)

# ==========================================
# RESHAPE FOR INVERSE SCALING
# ==========================================
pred_flat   = pred.reshape(-1, pred.shape[-1])
y_test_flat = y_test.reshape(-1, y_test.shape[-1])

# ==========================================
# CONVERT TO REAL SPEED VALUES (mph)
# ==========================================
pred_real = scaler.inverse_transform(pred_flat)
y_real    = scaler.inverse_transform(y_test_flat)

# ==========================================
# METRICS — RMSE, MAE, MAPE
# (assignment requires all three)
# ==========================================
mse  = mean_squared_error(y_real, pred_real)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_real, pred_real)

# MAPE — avoid divide-by-zero with small epsilon
mape = np.mean(
    np.abs((y_real - pred_real) / (np.abs(y_real) + 1e-5))
) * 100

print("\nREAL SCALE METRICS")
print(f"  RMSE : {rmse:.4f} mph")
print(f"  MAE  : {mae:.4f} mph")
print(f"  MAPE : {mape:.4f} %")

# ==========================================
# CONGESTION CLASSIFICATION METRICS
# ==========================================
from sklearn.metrics import accuracy_score, f1_score

def classify_speed(speed_mph):
    """Classify real-scale speed into congestion level."""
    if speed_mph < 25:
        return 2   # Heavy
    elif speed_mph < 50:
        return 1   # Medium
    else:
        return 0   # Free

y_class    = np.vectorize(classify_speed)(y_real)
pred_class = np.vectorize(classify_speed)(pred_real)

acc = accuracy_score(y_class.flatten(), pred_class.flatten())
f1  = f1_score(y_class.flatten(), pred_class.flatten(), average="weighted")

print(f"\nCONGESTION CLASSIFICATION")
print(f"  Accuracy : {acc:.4f}")
print(f"  F1 Score : {f1:.4f}")

# ==========================================
# RESHAPE BACK TO (samples, 2, n_sensors)
# ==========================================
pred_real = pred_real.reshape(pred.shape)
y_real    = y_real.reshape(y_test.shape)

# ==========================================
# SAMPLE OUTPUT — first sensor, 20 samples
# ==========================================
sensor_index = 0
num_samples  = 20

print(f"\nActual vs Predicted — Sensor {sensor_index} (first {num_samples} samples)\n")

for i in range(num_samples):
    actual    = y_real[i][0][sensor_index]
    predicted = pred_real[i][0][sensor_index]
    diff      = predicted - actual
    print(
        f"  Sample {i+1:02d} | "
        f"Actual: {actual:6.2f} mph | "
        f"Predicted: {predicted:6.2f} mph | "
        f"Diff: {diff:+.2f}"
    )

# ==========================================
# WORST SENSORS (error > 5 mph)
# ==========================================
errors    = np.abs(y_real - pred_real)
threshold = 5

sensor_error_count = np.sum(errors > threshold, axis=(0, 1))
top_sensors        = np.argsort(sensor_error_count)[-5:]

print(f"\nSensors with most errors > {threshold} mph:")
for s in top_sensors:
    print(f"  Sensor {s:03d} → {sensor_error_count[s]} times")

print(f"\nMax single error : {np.max(errors):.2f} mph")
print(f"Total errors > {threshold} mph : {np.sum(errors > threshold)}")