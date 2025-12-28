# phase4_predict_helper.py
import numpy as np
import joblib
import os

# 🔹 Get directory of this file (model folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_PATH  = os.path.join(BASE_DIR, "chlorophyll_model.pkl")


def load_artifacts():
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    return scaler, model


def predict_chlorophyll_from_params(
    temp, do, ph, sal, turb, cond
):
    scaler, model = load_artifacts()

    x = np.array([[temp, do, ph, sal, turb, cond]], dtype=float)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    return float(pred)


# 🔹 Direct test
if __name__ == "__main__":
    example_pred = predict_chlorophyll_from_params(
        temp=24.0,
        do=6.5,
        ph=8.0,
        sal=35.9,
        turb=5.3,
        cond=54.5
    )

    print("✅ Example predicted chlorophyll-a:", example_pred)
