Absolutely — here is a polished, professional, copy-paste-ready `README.md` version:

````markdown
# Employee Attrition ML Pipeline

![CI](https://github.com/stephens0n-0x/ml-attrition-pipeline/actions/workflows/ci.yml/badge.svg)

An end-to-end machine learning pipeline for predicting employee attrition using the IBM HR Analytics dataset.

The project trains and compares multiple classification models, addresses class imbalance with SMOTE, evaluates performance using imbalance-aware metrics, exposes predictions through a FastAPI endpoint, and generates plain-English model insights using a locally running LLM through Ollama.

No cloud LLM API is required.

---

## Overview

Employee attrition is a costly and difficult problem for organisations. The goal of this project is to predict which employees are at risk of leaving so that HR teams can better understand potential drivers of attrition and take proactive action.

The dataset is highly imbalanced: only **16.12%** of employees in the dataset left the company. A naive model that predicts every employee will stay would achieve **83.88% accuracy**, while providing no useful attrition detection.

To address this, the pipeline focuses on metrics that are more appropriate for imbalanced classification problems, including:

- F1 score
- Recall
- Precision
- ROC-AUC

The pipeline also uses **SMOTE** to balance the training data and **StratifiedKFold** cross-validation to preserve the original class distribution across validation folds.

---

## Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---------------------|---------:|----------:|-------:|---------:|--------:|
| Random Forest       |   0.7993 |    0.3636 | 0.3404 |   0.3516 |  0.7173 |
| Logistic Regression |   0.7857 |    0.3750 | 0.5106 |   0.4324 |  0.7092 |
| Decision Tree       |   0.6905 |    0.2250 | 0.3830 |   0.2835 |  0.4867 |

Random Forest achieved the highest cross-validation F1 score on SMOTE-augmented training data:

```text
Random Forest CV F1: 0.907 ± 0.013
````

However, Logistic Regression generalised better on the real held-out test set, achieving the highest test F1 score:

```text
Logistic Regression Test F1: 0.4324
Random Forest Test F1: 0.3516
```

This difference suggests that Random Forest may have overfit to synthetic SMOTE samples, while Logistic Regression produced more stable performance on unseen data.

---

## Ollama-Generated Insights

This project uses **Ollama** with **Llama 3.2** to generate local, plain-English interpretations of the model results.

### Model Comparison

Generated locally by Llama 3.2 via Ollama:

> Based on the provided metrics, the random forest classifier generalizes best to employee attrition predictions. Although it has lower precision and recall compared to logistic regression, its higher ROC-AUC score (0.7173) indicates better performance in distinguishing between actual and predicted positive class labels. Additionally, its cross-validation F1 scores suggest that it performs well on unseen data, outperforming both logistic regression and decision tree.

### Top Attrition Drivers

Random Forest feature importances interpreted by Llama 3.2:

> Employees are leaving due to a mismatch between their financial needs (MonthlyIncome) and the value they perceive in their job options (StockOptionLevel). Low levels of job satisfaction and environmental satisfaction indicate that employees feel undervalued and disconnected from their work environment. HR teams should prioritize addressing these drivers by offering competitive stock option packages, recognizing employee contributions, and fostering a positive work culture.

---

## Tech Stack

| Component         | Technology                                        |
| ----------------- | ------------------------------------------------- |
| Data manipulation | Pandas, NumPy                                     |
| Machine learning  | Scikit-learn                                      |
| Models            | Random Forest, Logistic Regression, Decision Tree |
| Class imbalance   | imbalanced-learn, SMOTE                           |
| Visualisation     | Matplotlib, Seaborn                               |
| Local LLM         | Ollama, Llama 3.2 3B                              |
| API               | FastAPI, Uvicorn                                  |
| Testing           | Pytest                                            |
| CI/CD             | GitHub Actions                                    |

---

## Project Structure

```text
ml-attrition-pipeline/
├── src/
│   ├── data_loader.py       # Dataset loading, EDA summary, and preprocessing
│   ├── train.py             # SMOTE, stratified CV, and model training
│   ├── evaluate.py          # Metrics, ROC curves, confusion matrices, feature importance
│   ├── ollama_insights.py   # Local LLM report generation
│   └── api.py               # FastAPI prediction endpoint
├── tests/
│   ├── test_data_loader.py
│   └── test_evaluate.py
├── configs/
│   └── config.yaml          # Centralised hyperparameter configuration
├── notebooks/
│   └── exploration.ipynb
├── .github/workflows/
│   └── ci.yml               # Runs tests on every push
└── main.py                  # End-to-end pipeline runner
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/stephens0n-0x/ml-attrition-pipeline.git
cd ml-attrition-pipeline
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Ollama Setup

To enable local LLM-generated insights, install Ollama from:

```text
https://ollama.com
```

Then pull the Llama 3.2 model:

```bash
ollama pull llama3.2
```

The pipeline can still run without Ollama by using the `--skip-ollama` flag.

---

## Usage

Run the full pipeline:

```bash
python main.py
```

Run the pipeline without Ollama insights:

```bash
python main.py --skip-ollama
```

Run tests:

```bash
pytest tests/ -v
```

Start the FastAPI prediction service:

```bash
uvicorn src.api:app --reload --app-dir .
```

Once the API is running, open the interactive documentation at:

```text
http://localhost:8000/docs
```

---

## Key Design Decisions

### Handling Class Imbalance with SMOTE

The dataset contains significantly more employees who stayed than employees who left. To address this imbalance, the pipeline applies SMOTE to the training data.

SMOTE creates synthetic minority-class examples by interpolating between existing minority-class samples. It is applied only to the training set to avoid data leakage.

### Stratified Cross-Validation

The pipeline uses StratifiedKFold cross-validation to preserve the original attrition ratio across each fold.

This is important because standard KFold can produce validation folds with too few minority-class samples, especially in imbalanced datasets.

### Metrics Beyond Accuracy

Accuracy alone is misleading for this problem because the majority class dominates the dataset.

Instead, the project prioritises:

* **F1 score** — balances precision and recall
* **Recall** — measures how many actual attrition cases are detected
* **Precision** — measures how many predicted attrition cases are correct
* **ROC-AUC** — measures the model's ability to distinguish between classes across thresholds

### Local LLM Insights

The project uses Ollama to generate natural-language explanations of model performance and feature importance.

This allows model interpretation to happen locally without sending data to an external API.

### Config-Driven Pipeline

All key hyperparameters are stored in:

```text
configs/config.yaml
```

This makes it easier to adjust model settings, train/test splits, random seeds, and Ollama configuration without changing the core pipeline code.

---

## Dataset

This project uses the IBM HR Analytics Employee Attrition dataset.

The dataset contains:

* 1,470 employee records
* 35 features
* Demographic attributes
* Job role information
* Satisfaction scores
* Compensation-related variables
* Attrition labels

Dataset source: [IBM Employee Attrition AIF360 Repository](https://github.com/IBM/employee-attrition-aif360)


```
