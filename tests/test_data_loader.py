import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, "src")
from data_loader import load_config, get_data_summary, preprocess


@pytest.fixture
def sample_df():
    """Minimal synthetic DataFrame mirroring the IBM HR dataset schema."""
    return pd.DataFrame({
        "Age": [35, 28, 45, 52, 31],
        "Attrition": ["Yes", "No", "No", "Yes", "No"],
        "BusinessTravel": ["Travel_Rarely", "Non-Travel", "Travel_Frequently", "Travel_Rarely", "Non-Travel"],
        "DailyRate": [800, 1200, 950, 600, 1100],
        "Department": ["Sales", "Research & Development", "Human Resources", "Sales", "Research & Development"],
        "DistanceFromHome": [5, 20, 3, 15, 8],
        "Education": [3, 4, 2, 5, 3],
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "MonthlyIncome": [5000, 8000, 6500, 4000, 9000],
        "OverTime": ["Yes", "No", "Yes", "No", "No"],
        "YearsAtCompany": [3, 7, 12, 1, 9],
        "EmployeeCount": [1, 1, 1, 1, 1],       # constant column — should be dropped
        "StandardHours": [80, 80, 80, 80, 80],   # constant column — should be dropped
    })


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


def test_load_config(config):
    assert "data" in config
    assert "models" in config
    assert "ollama" in config
    assert "api" in config


def test_get_data_summary(sample_df):
    summary = get_data_summary(sample_df)
    assert summary["n_rows"] == 5
    assert summary["n_columns"] == len(sample_df.columns)
    assert "attrition_rate_pct" in summary
    assert summary["missing_values"] == 0


def test_preprocess_drops_constant_columns(sample_df, config):
    X, y = preprocess(sample_df, config)
    assert "EmployeeCount" not in X.columns
    assert "StandardHours" not in X.columns


def test_preprocess_encodes_target(sample_df, config):
    X, y = preprocess(sample_df, config)
    assert set(y.unique()).issubset({0, 1})


def test_preprocess_no_object_columns(sample_df, config):
    X, y = preprocess(sample_df, config)
    object_cols = X.select_dtypes(include="object").columns.tolist()
    assert len(object_cols) == 0


def test_preprocess_correct_split(sample_df, config):
    X, y = preprocess(sample_df, config)
    assert len(X) == len(y)
    assert "Attrition" not in X.columns