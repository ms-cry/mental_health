from typing import List, Dict, Any

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from src.preprocessing.preprocessor import preprocess_texts
from src.config.config import LABEL_MAP


def evaluate_model(
    model,
    vectorizer,
    texts: List[str],
    y_true: List[int],
) -> Dict[str, Any]:
    """
    Evaluate model performance.
    """

    processed = preprocess_texts(texts)
    X = vectorizer.transform(processed)

    y_pred = model.predict(X)

    labels = sorted(LABEL_MAP.keys())
    target_names = [LABEL_MAP[l] for l in labels]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0
        ),
    }


def print_report(metrics: Dict[str, Any]) -> None:
    """
    Print evaluation metrics.
    """

    print("\n=== Model Evaluation ===")
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"F1 (Macro)    : {metrics['f1_macro']:.4f}")
    print(f"F1 (Weighted) : {metrics['f1_weighted']:.4f}")

    print("\nClassification Report:")
    print(metrics["classification_report"])

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])