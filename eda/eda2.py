# phase2_eda_split.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

CLEAN_CSV = "datasets/clean_water_quality.csv"
SPLIT_NPY_PREFIX = "split_"

def run_eda_and_split():
    # 🔹 Read data
    df = pd.read_csv(CLEAN_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(df.head())
    print(df.describe())

    # 🔹 Plot chlorophyll-a
    plt.figure(figsize=(12, 4))
    plt.plot(df['date'], df['chlorophyll_a'])
    plt.title("Chlorophyll-a over time")
    plt.xlabel("Date")
    plt.ylabel("Chlorophyll-a")
    plt.tight_layout()
    plt.show()

    # 🔹 Correlation heatmap (FIXED column names)
    corr = df[['chlorophyll_a', 'temperature', 'dissolved_oxygen',
               'pH', 'Salinity', 'Turbidity', 'conductivity']].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    # 🔹 Features & target
    feature_cols = ['temperature', 'dissolved_oxygen',
                    'pH', 'Salinity', 'Turbidity', 'conductivity']
    target_col = 'chlorophyll_a'

    X = df[feature_cols].values
    y = df[target_col].to_numpy()

    # 🔹 Chronological split
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # 🔹 CURRENT WORKING DIRECTORY (important part)
    cwd = os.getcwd()
    print("Current working directory:", cwd)

    # (Optional but recommended)
    split_dir = os.path.join(cwd, "splits")
    os.makedirs(split_dir, exist_ok=True)

    # 🔹 Save files explicitly in current directory
    np.save(os.path.join(split_dir, "split_X_train.npy"), X_train)
    np.save(os.path.join(split_dir, "split_y_train.npy"), y_train)
    np.save(os.path.join(split_dir, "split_X_val.npy"), X_val)
    np.save(os.path.join(split_dir, "split_y_val.npy"), y_val)
    np.save(os.path.join(split_dir, "split_X_test.npy"), X_test)
    np.save(os.path.join(split_dir, "split_y_test.npy"), y_test)

    print("✅ Saved split files inside:", split_dir)

if __name__ == "__main__":
    run_eda_and_split()
