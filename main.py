import sys
import argparse
from pathlib import Path

sys.path.insert(0, "src")

from data_loader import load_config, fetch_dataset, get_data_summary, preprocess
from train import train_all, save_models
from evaluate import evaluate_all, get_feature_importance, plot_roc_curves, plot_confusion_matrices, plot_feature_importance, build_summary
from ollama_insights import check_ollama_available, generate_full_report


def run_pipeline(skip_ollama: bool = False):
    """
    Execute the full ML pipeline end to end:
    data loading -> preprocessing -> training -> evaluation -> insights.
    """
    config = load_config()

    # ── Step 1: Load and inspect data ─────────────────────────────────────────
    print("\n[pipeline] Step 1: Loading data...")
    df = fetch_dataset(config)
    summary = get_data_summary(df)

    print(f"\n[pipeline] Dataset overview:")
    print(f"           Rows:           {summary['n_rows']}")
    print(f"           Columns:        {summary['n_columns']}")
    print(f"           Missing values: {summary['missing_values']}")
    print(f"           Attrition rate: {summary['attrition_rate_pct']}%")
    print(f"           Class counts:   {summary['attrition_counts']}")

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    print("\n[pipeline] Step 2: Preprocessing...")
    X, y = preprocess(df, config)
    print(f"[pipeline] Feature matrix: {X.shape}")

    # ── Step 3: Train ─────────────────────────────────────────────────────────
    print("\n[pipeline] Step 3: Training models...")
    results = train_all(X, y, config)
    save_models(results, config)

    # ── Step 4: Evaluate ──────────────────────────────────────────────────────
    print("\n[pipeline] Step 4: Evaluating models...")
    metrics = evaluate_all(results)

    importance_df = get_feature_importance(results, X.columns.tolist())

    plot_roc_curves(metrics, results["y_test"])
    plot_confusion_matrices(metrics, results["y_test"])
    plot_feature_importance(importance_df)

    eval_summary = build_summary(metrics, results["cv_scores"], importance_df)

    # ── Step 5: Ollama insights ───────────────────────────────────────────────
    if skip_ollama:
        print("\n[pipeline] Step 5: Skipping Ollama insights (--skip-ollama flag set)")
    else:
        host = config["ollama"]["host"]
        if check_ollama_available(host):
            print("\n[pipeline] Step 5: Generating Ollama insights...")
            generate_full_report(eval_summary, config)
        else:
            print("\n[pipeline] Step 5: Ollama not running — skipping insights.")
            print("           Start Ollama and re-run to generate natural language analysis.")

    print("\n[pipeline] Done. Outputs saved to outputs/")
    return eval_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBM HR Attrition ML Pipeline")
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip Ollama insights generation (useful if Ollama is not running)"
    )
    args = parser.parse_args()
    run_pipeline(skip_ollama=args.skip_ollama)