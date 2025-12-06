"""
data_prep.py
- Loads dataset from HF dataset repo (or local file if present)
- Cleans data: drop ID cols, trim strings, normalize Gender, IQR capping for certain numerics
- Creates stratified train/test split and saves CSVs in tourism_project/data/
- Uploads train.csv and test.csv back to HF dataset repo
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, upload_file, create_repo

HF_USERNAME = os.getenv("HF_USERNAME", "sathishaiuse")
DATASET_NAME = os.getenv("HF_DATASET", "wellness-tourism-dataset-final")
REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"

LOCAL_DIR = Path("tourism_project/data")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_RAW = LOCAL_DIR / "tourism.csv"
CLEAN_PATH = LOCAL_DIR / "cleaned_full.csv"
TRAIN_PATH = LOCAL_DIR / "train.csv"
TEST_PATH = LOCAL_DIR / "test.csv"

def iqr_cap(series, factor=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return series.clip(lower=lower, upper=upper)

def normalize_gender(s):
    s = str(s).strip().lower()
    if s.startswith("m"):
        return "Male"
    if s.startswith("f"):
        return "Female"
    return "Other"

def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not found in environment.")

    # Load source data: prefer HF dataset if not present locally
    if LOCAL_RAW.exists():
        print("Loading raw dataset from local file:", LOCAL_RAW)
        df = pd.read_csv(LOCAL_RAW)
    else:
        print("Loading dataset from HF datasets:", REPO_ID)
        ds = load_dataset(REPO_ID, data_files={"all": "tourism.csv"}, split="all")
        df = ds.to_pandas()
        print("Loaded from HF, shape:", df.shape)

    # Drop id-like columns
    for c in ["Unnamed: 0", "CustomerID"]:
        if c in df.columns:
            df = df.drop(columns=c)

    # Trim strings
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Normalize gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].apply(normalize_gender)

    # Fill missing if any (fallback)
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    # IQR capping on candidate numeric columns if present
    candidates = ["DurationOfPitch", "NumberOfTrips", "MonthlyIncome", "Age", "NumberOfFollowups"]
    for c in candidates:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = iqr_cap(df[c])

    # Ensure target is integer
    if "ProdTaken" in df.columns:
        df["ProdTaken"] = df["ProdTaken"].astype(int)

    # Save cleaned full
    df.to_csv(CLEAN_PATH, index=False)
    print("Saved cleaned dataset to:", CLEAN_PATH)

    # Stratified split
    if "ProdTaken" not in df.columns:
        raise ValueError("Target column 'ProdTaken' not found in dataset.")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["ProdTaken"])
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print("Saved train/test to:", TRAIN_PATH, TEST_PATH)
    print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

    # Upload to HF dataset repo
    api = HfApi()
    try:
        create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print("Warning creating dataset repo:", e)

    for local_path, remote_name in [(CLEAN_PATH, "cleaned_full.csv"), (TRAIN_PATH, "train.csv"), (TEST_PATH, "test.csv")]:
        try:
            print(f"Uploading {local_path} -> {REPO_ID}/{remote_name}")
            upload_file(path_or_fileobj=str(local_path), path_in_repo=remote_name, repo_id=REPO_ID, repo_type="dataset", token=token)
            print("Uploaded:", remote_name)
        except Exception as e:
            print("Upload failed for", remote_name, ":", type(e).__name__, e)

if __name__ == "__main__":
    main()
