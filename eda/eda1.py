# phase1_load_clean.py
import pandas as pd

INPUT_CSV = "datasets/final_water_quality_chlorophyll_dataset.csv"
OUTPUT_CSV = "datasets/clean_water_quality.csv"

def load_and_clean():
    # 1) Load
    df = pd.read_csv(INPUT_CSV)

    # 2) Parse datetime and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 3) Basic cleaning
    # Drop exact duplicates
    df = df.drop_duplicates(subset=['date'])

    # Forward/backward fill small gaps (you already did interpolation before,
    # this is just a safety net)
    df = df.ffill().bfill()

    # 4) Save cleaned data
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved cleaned data to {OUTPUT_CSV}")
    return df

if __name__ == "__main__":
    df_clean = load_and_clean()
    print(df_clean.head())
    print(df_clean.info())
