import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config.config import (
    LR_C,
    LR_MAX_ITER,
    MODEL_PATH,
    RANDOM_STATE,
    VECTORIZER_PATH,
)
from src.data.data_loader import load_dataset
from src.evaluation.metrics import evaluate_model, print_report
from src.features.tfidf_vectorizer import build_vectorizer
from src.preprocessing.preprocessor import preprocess_texts


def train_model(data_path: str) -> None:
    """
    Train Logistic Regression model using TF-IDF features.
    """

    df = load_dataset(data_path)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    print(f"[INFO] Total samples: {len(texts)}")

    texts = preprocess_texts(texts)

    if not texts or all(not text.strip() for text in texts):
        raise ValueError(
            "All texts became empty after preprocessing. "
            "Check preprocessor.py"
        )

    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)

    print(f"[INFO] TF-IDF shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(
        f"[INFO] Train size: {len(y_train)} | "
        f"Test size: {len(y_test)}"
    )

    model = LogisticRegression(
        max_iter=LR_MAX_ITER,
        C=LR_C,
        solver="lbfgs",
    )

    model.fit(X_train, y_train)

    print("[INFO] Model training completed")

    metrics = evaluate_model(model, X_test, y_test)
    print_report(metrics)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(
        f"[INFO] Model saved -> {MODEL_PATH}\n"
        f"[INFO] Vectorizer saved -> {VECTORIZER_PATH}"
    )


if __name__ == "__main__":
    train_model("data/raw/dataset.csv")
