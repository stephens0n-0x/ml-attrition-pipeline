import pandas as pd
import numpy as np
import yaml
import requests
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_dataset(config: dict) -> pd.DataFrame:
    """
    Download the IBM HR Analytics dataset if not already cached locally.
    Returns raw DataFrame with original column names and dtypes intact.
    """
    cache_path = Path("data/hr_attrition_raw.csv")

    if cache_path.exists():
        print("[data] Loading cached dataset...")
        return pd.read_csv(cache_path)

    print("[data] Downloading IBM HR Analytics dataset...")
    response = requests.get(config["data"]["url"], timeout=30)
    response.raise_for_status()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(response.content)

    df = pd.read_csv(cache_path)
    print(f"[data] Downloaded {len(df)} rows, {len(df.columns)} columns")
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Compute descriptive statistics and data quality indicators.
    Used in EDA to understand the dataset before any preprocessing.
    """
    target = "Attrition"
    attrition_counts = df[target].value_counts()
    attrition_rate = (attrition_counts.get("Yes", 0) / len(df)) * 100

    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "missing_values": df.isnull().sum().sum(),
        "attrition_rate_pct": round(attrition_rate, 2),
        "attrition_counts": attrition_counts.to_dict(),
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": df.select_dtypes(include="object").columns.tolist(),
    }


def preprocess(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean and encode the raw DataFrame for model training.

    Drops constant columns (EmployeeCount, StandardHours, Over18) that carry
    no predictive signal. Encodes binary categoricals as 0/1 and applies
    LabelEncoder to remaining object columns.

    Returns feature matrix X and binary target y.
    """
    df = df.copy()

    # Columns with a single unique value contribute zero information
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df.drop(columns=constant_cols, inplace=True)

    # Encode target: Yes -> 1, No -> 0
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)

    # Binary string columns to 0/1
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() == 2:
            df[col] = df[col].map(binary_map).fillna(df[col])

    # Encode remaining categoricals
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))

    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    return X, y