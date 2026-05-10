import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


def compute_metrics(model, X_test, y_test, model_name: str) -> dict:
    """
    Compute classification metrics for a single trained model.

    Reports precision, recall, and F1 alongside accuracy and AUC-ROC.
    For an imbalanced dataset like this one, AUC-ROC and F1 are the
    primary indicators of model quality — accuracy alone is misleading.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def evaluate_all(results: dict) -> list[dict]:
    """
    Evaluate all trained models and return sorted metrics.
    Sorted by ROC-AUC descending so the best model is always first.
    """
    X_test = results["X_test"]
    y_test = results["y_test"]

    metrics = []
    for name, model in results["models"].items():
        m = compute_metrics(model, X_test, y_test, name)
        metrics.append(m)
        print(f"\n[eval] {name}")
        print(f"       Accuracy:  {m['accuracy']}")
        print(f"       Precision: {m['precision']}")
        print(f"       Recall:    {m['recall']}")
        print(f"       F1:        {m['f1']}")
        print(f"       ROC-AUC:   {m['roc_auc']}")

    return sorted(metrics, key=lambda x: x["roc_auc"], reverse=True)


def get_feature_importance(results: dict, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances from tree-based models.
    Returns a DataFrame sorted by Random Forest importance descending.
    Used to identify the top drivers of employee attrition.
    """
    importance_data = {}

    for name, model in results["models"].items():
        if hasattr(model, "feature_importances_"):
            importance_data[name] = model.feature_importances_

    if not importance_data:
        return pd.DataFrame()

    df = pd.DataFrame(importance_data, index=feature_names)
    df = df.sort_values("random_forest", ascending=False)
    return df


def plot_roc_curves(metrics: list[dict], y_test, output_dir: str = "outputs"):
    """Plot ROC curves for all models on a single axes for direct comparison."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for m in metrics:
        fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
        ax.plot(fpr, tpr, label=f"{m['model']} (AUC = {m['roc_auc']})")

    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Attrition Classifiers")
    ax.legend()
    ax.grid(alpha=0.3)

    path = Path(output_dir) / "roc_curves.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[eval] ROC curves saved to {path}")


def plot_confusion_matrices(metrics: list[dict], y_test, output_dir: str = "outputs"):
    """Plot confusion matrices side by side for all models."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        cm = confusion_matrix(y_test, m["y_pred"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Stay", "Leave"],
            yticklabels=["Stay", "Leave"],
            ax=ax,
        )
        ax.set_title(m["model"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    path = Path(output_dir) / "confusion_matrices.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[eval] Confusion matrices saved to {path}")


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15, output_dir: str = "outputs"):
    """Plot top N features by Random Forest importance."""
    if importance_df.empty:
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    top["random_forest"].plot(kind="barh", ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Features — Random Forest")
    ax.grid(axis="x", alpha=0.3)

    path = Path(output_dir) / "feature_importance.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[eval] Feature importance saved to {path}")


def build_summary(metrics: list[dict], cv_scores: dict, importance_df: pd.DataFrame) -> dict:
    """
    Assemble a structured summary dict for the Ollama insights generator.
    Keeps the interface between evaluate.py and ollama_insights.py clean.
    """
    best = metrics[0]
    top_features = importance_df.head(5).index.tolist() if not importance_df.empty else []

    return {
        "best_model": best["model"],
        "best_roc_auc": best["roc_auc"],
        "best_f1": best["f1"],
        "all_metrics": [
            {k: v for k, v in m.items() if k not in ("y_pred", "y_prob")}
            for m in metrics
        ],
        "cv_scores": cv_scores,
        "top_attrition_drivers": top_features,
    }