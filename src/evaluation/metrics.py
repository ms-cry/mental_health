from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.config.config import LABEL_MAP


def evaluate_model(
    model,
    X_test,
    y_test,
) -> Dict[str, Any]:
    """
    Evaluate trained model using test features directly.

    Parameters:
    - model: trained Logistic Regression model
    - X_test: TF-IDF transformed test features
    - y_test: true labels

    Returns:
    - Dictionary containing evaluation metrics
    """

    y_pred = model.predict(X_test)

    labels = sorted(LABEL_MAP.keys())
    target_names = [LABEL_MAP[label] for label in labels]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),

        "precision_macro": precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1_macro": f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "f1_weighted": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=labels
        ),

        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0
        ),
    }

    return metrics


def print_report(metrics: Dict[str, Any]) -> None:
    """
    Print formatted evaluation report
    """

    print("\n========== MODEL EVALUATION ==========\n")

    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Precision Macro  : {metrics['precision_macro']:.4f}")
    print(f"Recall Macro     : {metrics['recall_macro']:.4f}")
    print(f"F1 Score Macro   : {metrics['f1_macro']:.4f}")
    print(f"F1 Score Weighted: {metrics['f1_weighted']:.4f}")

    print("\n========== CLASSIFICATION REPORT ==========\n")
    print(metrics["classification_report"])

    print("\n========== CONFUSION MATRIX ==========\n")
    print(metrics["confusion_matrix"])