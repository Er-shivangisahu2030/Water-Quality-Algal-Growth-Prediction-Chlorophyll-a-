# phase3_train_baseline.py
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 🔹 Base directories
BASE_DIR = r"E:\PROJECT FILE\PROJECT 6- WATER PURITY PREDICTION DUE TO ALGAE GROWTH"
SPLIT_DIR = os.path.join(BASE_DIR, "splits")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# 🔹 Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH  = os.path.join(MODEL_DIR, "chlorophyll_model.pkl")


def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")


def train_and_save_models():
    # 🔹 Load splits (FIXED)
    X_train = np.load(os.path.join(SPLIT_DIR, "split_X_train.npy"))
    y_train = np.load(os.path.join(SPLIT_DIR, "split_y_train.npy"))
    X_val   = np.load(os.path.join(SPLIT_DIR, "split_X_val.npy"))
    y_val   = np.load(os.path.join(SPLIT_DIR, "split_y_val.npy"))
    X_test  = np.load(os.path.join(SPLIT_DIR, "split_X_test.npy"))
    y_test  = np.load(os.path.join(SPLIT_DIR, "split_y_test.npy"))

    # 🔹 Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # 🔹 Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)

    y_val_pred_lin  = lin_reg.predict(X_val_scaled)
    y_test_pred_lin = lin_reg.predict(X_test_scaled)

    print_metrics("LinearReg VAL",  y_val,  y_val_pred_lin)
    print_metrics("LinearReg TEST", y_test, y_test_pred_lin)

    # 🔹 Random Forest
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)

    y_val_pred_rf  = rf.predict(X_val_scaled)
    y_test_pred_rf = rf.predict(X_test_scaled)

    print_metrics("RandomForest VAL",  y_val,  y_val_pred_rf)
    print_metrics("RandomForest TEST", y_test, y_test_pred_rf)

    # 🔹 Model selection
    mae_lin = mean_absolute_error(y_val, y_val_pred_lin)
    mae_rf  = mean_absolute_error(y_val, y_val_pred_rf)

    best_model = rf if mae_rf <= mae_lin else lin_reg
    best_name  = "RandomForest" if mae_rf <= mae_lin else "LinearRegression"

    print(f"✅ Best model based on VAL MAE: {best_name}")

    # 🔹 Save artifacts
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(best_model, MODEL_PATH)

    print("✅ Saved scaler to:", SCALER_PATH)
    print("✅ Saved model  to:", MODEL_PATH)


if __name__ == "__main__":
    train_and_save_models()
