import joblib
from pathlib import Path
from typing import List, Dict

from src.preprocessing.preprocessor import preprocess_texts
from src.config.config import (
    MODEL_PATH,
    VECTORIZER_PATH,
    LABEL_MAP,
    INTERPRETATIONS,
)


def load_model():
    """
    Load trained model and vectorizer.
    """
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError("Model not found. Train first.")

    if not Path(VECTORIZER_PATH).exists():
        raise FileNotFoundError("Vectorizer not found. Train first.")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def predict(texts: List[str]) -> List[Dict]:
    """
    Predict labels for multiple texts.
    """
    if not texts:
        raise ValueError("Input text list is empty.")

    model, vectorizer = load_model()

    processed = preprocess_texts(texts)
    X = vectorizer.transform(processed)

    preds = model.predict(X)

    results = []

    for text, pred in zip(texts, preds):
        label = LABEL_MAP.get(pred, "normal")
        interpretation = INTERPRETATIONS[label]

        results.append({
            "text": text,
            "label_id": int(pred),
            "label": label,
            "interpretation": interpretation
        })

    return results


def predict_single(text: str) -> Dict:
    """
    Predict a single text.
    """
    return predict([text])[0]