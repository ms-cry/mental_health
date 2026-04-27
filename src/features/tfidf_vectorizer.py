import logging
import pickle
from pathlib import Path
from typing import Iterable

from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)

logger = logging.getLogger(__name__)


def build_vectorizer(
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    max_features: int = TFIDF_MAX_FEATURES,
) -> TfidfVectorizer:
    """
    Build optimized TF-IDF vectorizer for sentiment classification.
    """

    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=1,              # keep rare but important words
        max_df=0.9,            # remove overly common words
        lowercase=True,
        stop_words=None,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b\w+\b",  # allow short tokens like "i", "ok"
    )

    logger.info(
        "Vectorizer initialized | ngram_range=%s | max_features=%d",
        ngram_range,
        max_features,
    )

    return vec


def fit_vectorizer(vectorizer: TfidfVectorizer, texts: Iterable[str]) -> spmatrix:
    matrix = vectorizer.fit_transform(texts)

    logger.info(
        "Vectorizer fitted | vocab_size=%d | shape=%s",
        len(vectorizer.vocabulary_),
        matrix.shape,
    )

    return matrix


def transform_texts(vectorizer: TfidfVectorizer, texts: Iterable[str]) -> spmatrix:
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)

    logger.info("Vectorizer saved → %s", path)


def load_vectorizer(path: str | Path) -> TfidfVectorizer:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Vectorizer not found: {path}")

    with open(path, "rb") as f:
        vec = pickle.load(f)

    logger.info("Vectorizer loaded ← %s", path)
    return vec