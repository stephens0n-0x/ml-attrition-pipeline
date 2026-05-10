import os
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_models(config: dict) -> dict:
    """
    Instantiate all classifiers from config parameters.
    Keeping model construction here makes swapping or adding models trivial.
    """
    cfg = config["models"]
    return {
        "random_forest": RandomForestClassifier(**cfg["random_forest"]),
        "logistic_regression": LogisticRegression(**cfg["logistic_regression"]),
        "decision_tree": DecisionTreeClassifier(**cfg["decision_tree"]),
    }


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance.

    Rather than simply duplicating minority-class samples, SMOTE generates
    synthetic examples by interpolating between existing minority samples in
    feature space. This produces a more generalizable training distribution
    than random oversampling.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"[train] SMOTE applied: {y_train.value_counts().to_dict()} -> {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled


def train_all(X, y, config: dict) -> dict:
    """
    Split data, apply SMOTE, scale features, and train all classifiers.

    Uses StratifiedKFold cross-validation to ensure class distribution
    is preserved across folds — critical for imbalanced datasets where
    random splits can produce folds with no minority-class samples.

    Returns a results dict containing trained models, scaler, split data,
    and cross-validation scores for each classifier.
    """
    random_state = config["data"]["random_state"]
    test_size = config["data"]["test_size"]
    n_folds = config["training"]["cross_val_folds"]

    # Stratified split preserves the attrition ratio in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train, y_train = apply_smote(X_train, y_train, random_state)

    # Logistic regression requires feature scaling; applying to all models
    # is harmless for tree-based models and keeps the pipeline consistent
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models(config)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    results = {
        "scaler": scaler,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "models": {},
        "cv_scores": {},
    }

    for name, model in models.items():
        print(f"[train] Training {name}...")
        model.fit(X_train_scaled, y_train)

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1")
        results["models"][name] = model
        results["cv_scores"][name] = {
            "mean_f1": round(cv_scores.mean(), 4),
            "std_f1": round(cv_scores.std(), 4),
        }
        print(f"[train] {name} CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results


def save_models(results: dict, config: dict):
    """Persist trained models and scaler to disk for use by the API."""
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(results["scaler"], output_dir / "scaler.pkl")

    for name, model in results["models"].items():
        joblib.dump(model, output_dir / f"{name}.pkl")

    print(f"[train] Models saved to {output_dir}/")