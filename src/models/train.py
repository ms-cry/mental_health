import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data.data_loader import load_dataset
from src.preprocessing.preprocessor import preprocess_texts
from src.features.tfidf_vectorizer import build_vectorizer
from src.config.config import (
    MODEL_PATH,
    VECTORIZER_PATH,
    LR_MAX_ITER,
    LR_C,
    RANDOM_STATE,
)


def train_model(data_path: str) -> None:
    """
    Train Logistic Regression model using TF-IDF features.
    """

    # 1. Load data
    df = load_dataset(data_path)

    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # 2. Preprocess
    texts = preprocess_texts(texts)

    # 3. Vectorize
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # 5. Train model
    model = LogisticRegression(
        max_iter=LR_MAX_ITER,
        C=LR_C,
        solver="lbfgs",
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # 6. Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("[INFO] Model and vectorizer saved successfully.")


if __name__ == "__main__":
    train_model("data/raw/dataset.csv")