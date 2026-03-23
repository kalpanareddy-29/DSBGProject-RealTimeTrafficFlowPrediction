# ==========================================
# IMPORTS
# ==========================================
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    Attention, Flatten, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ==========================================
# PATHS — relative, works on any machine
# ==========================================
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================
print("Loading data...")

df = pd.read_csv(DATA_PATH)
df.rename(columns={"index": "timestamp"}, inplace=True, errors="ignore")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# ==========================================
# PREPARE DATA
# ==========================================
print("Preparing data...")
data = df.drop(columns=["timestamp"]).values

# ==========================================
# NORMALIZATION
# ==========================================
print("Scaling...")
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.save"))
print(f"Scaler saved to models/scaler.save")

# ==========================================
# CREATE SEQUENCES
# ==========================================
def create_sequences_multi(data, window=12, future_steps=[3, 6]):
    X, y = [], []
    for i in range(len(data) - window - max(future_steps)):
        X.append(data[i:i+window])
        future = [data[i+window+step-1] for step in future_steps]
        y.append(np.array(future))
    return np.array(X), np.array(y)

X, y = create_sequences_multi(data_scaled, 12)
print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================================
# SPLIT DATA
# ==========================================
train_size = int(0.7 * len(X))
val_size   = int(0.15 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]
X_val   = X[train_size:train_size + val_size]
y_val   = y[train_size:train_size + val_size]
X_test  = X[train_size + val_size:]
y_test  = y[train_size + val_size:]

print(f"\nSplit Done")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)

# ==========================================
# BUILD MODEL
# ==========================================
print("\nBuilding model...")

inputs = Input(shape=(12, X.shape[2]))

x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
x = Dropout(0.3)(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.3)(x)

attention = Attention()([x, x])
x = Flatten()(attention)

x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)

output = Dense(2 * X.shape[2])(x)
output = Reshape((2, X.shape[2]))(output)

model = Model(inputs, output)

# ==========================================
# COMPILE
# ==========================================
model.compile(
    optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
    loss="huber",
    metrics=["mae"]
)

model.summary()

# ==========================================
# CALLBACKS — saved to models/ folder
# ==========================================
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_model.keras"),
        save_best_only=True,
        verbose=1
    )
]

# ==========================================
# TRAIN
# ==========================================
print("\nTraining...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    callbacks=callbacks
)

# ==========================================
# EVALUATION
# ==========================================
print("\nEvaluating...")

pred = model.predict(X_test)

mae  = mean_absolute_error(y_test.reshape(-1), pred.reshape(-1))
rmse = np.sqrt(mean_squared_error(y_test.reshape(-1), pred.reshape(-1)))
mape = np.mean(
    np.abs((y_test.reshape(-1) - pred.reshape(-1)) /
           (np.abs(y_test.reshape(-1)) + 1e-5))
) * 100

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f} %")

# ==========================================
# CLASSIFICATION
# ==========================================
def classify(speed):
    if speed < 0.3:
        return 0
    elif speed < 0.6:
        return 1
    else:
        return 2

y_test_class = np.vectorize(classify)(y_test)
pred_class   = np.vectorize(classify)(pred)

print("Accuracy:", accuracy_score(y_test_class.flatten(), pred_class.flatten()))
print("F1 Score:", f1_score(y_test_class.flatten(), pred_class.flatten(), average="weighted"))

# ==========================================
# SAVE ALL FILES TO models/ FOLDER
# ==========================================
model.save(os.path.join(MODEL_DIR, "traffic_bilstm_attention.keras"))
print(f"Model saved    → models/traffic_bilstm_attention.keras")

np.save(os.path.join(MODEL_DIR, "X_test.npy"), X_test)
print(f"X_test saved   → models/X_test.npy")

np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_test)
print(f"y_test saved   → models/y_test.npy")

print("\nAll files saved to models/ folder!")
