import json
import requests
from typing import Optional


def _call_ollama(prompt: str, model: str, host: str) -> str:
    """
    Send a prompt to the local Ollama server and return the response text.
    Uses the /api/generate endpoint with stream=False for a single blocking response.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
        },
    }

    response = requests.post(
        f"{host}/api/generate",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def generate_model_comparison(summary: dict, config: dict) -> str:
    """
    Ask the local LLM to compare classifier performance in plain English.
    The prompt is structured to constrain the response to factual interpretation
    of the provided metrics rather than general ML commentary.
    """
    model = config["ollama"]["model"]
    host = config["ollama"]["host"]

    metrics_text = "\n".join([
        f"- {m['model']}: Accuracy={m['accuracy']}, Precision={m['precision']}, "
        f"Recall={m['recall']}, F1={m['f1']}, ROC-AUC={m['roc_auc']}"
        for m in summary["all_metrics"]
    ])

    cv_text = "\n".join([
        f"- {name}: mean F1={scores['mean_f1']} ± {scores['std_f1']}"
        for name, scores in summary["cv_scores"].items()
    ])

    prompt = f"""You are a senior data scientist reviewing employee attrition model results.
Based only on the metrics below, write a concise 3-4 sentence comparison of the three classifiers.
Focus on which model generalizes best and why, referencing specific numbers.
Do not add general ML advice — only interpret these specific results.

Classifier metrics on held-out test set:
{metrics_text}

Cross-validation F1 scores (5-fold, training set):
{cv_text}

Best model overall: {summary['best_model']} (ROC-AUC: {summary['best_roc_auc']}, F1: {summary['best_f1']})

Write the comparison now:"""

    print("[ollama] Generating model comparison...")
    return _call_ollama(prompt, model, host)


def generate_attrition_drivers(summary: dict, config: dict) -> str:
    """
    Ask the local LLM to interpret the top feature importances as business insights.
    Framed as an HR advisory summary rather than a technical ML explanation.
    """
    model = config["ollama"]["model"]
    host = config["ollama"]["host"]

    features_text = ", ".join(summary["top_attrition_drivers"])

    prompt = f"""You are a senior data scientist advising an HR department on employee retention.
The following features were identified as the top predictors of employee attrition
by a Random Forest classifier trained on IBM HR data:

Top attrition drivers: {features_text}

Write a concise 3-4 sentence business-oriented summary explaining what these factors
suggest about why employees leave and what HR teams should focus on.
Be specific to these features — do not give generic HR advice.

Write the summary now:"""

    print("[ollama] Generating attrition driver insights...")
    return _call_ollama(prompt, model, host)


def generate_full_report(summary: dict, config: dict) -> dict:
    """
    Generate a complete insights report combining model comparison and attrition drivers.
    Returns a structured dict that can be logged, saved, or served via the API.
    """
    report = {
        "model_comparison": generate_model_comparison(summary, config),
        "attrition_drivers": generate_attrition_drivers(summary, config),
        "best_model": summary["best_model"],
        "best_roc_auc": summary["best_roc_auc"],
        "top_features": summary["top_attrition_drivers"],
    }

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(report["model_comparison"])
    print("\n" + "=" * 60)
    print("ATTRITION DRIVERS")
    print("=" * 60)
    print(report["attrition_drivers"])

    return report


def check_ollama_available(host: str) -> bool:
    """
    Verify Ollama server is reachable before attempting generation.
    Allows the pipeline to degrade gracefully if Ollama is not running.
    """
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False