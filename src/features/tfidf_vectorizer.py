import logging
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
    """Build optimized TF-IDF vectorizer for sentiment classification."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=1,
        max_df=0.9,
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"\b\w+\b",
    )


def fit_vectorizer(vectorizer: TfidfVectorizer, texts: Iterable[str]) -> spmatrix:
    """Fit vectorizer and return transformed features."""
    matrix = vectorizer.fit_transform(texts)
    logger.info("Vectorizer fitted | vocab_size=%d | shape=%s", len(vectorizer.vocabulary_), matrix.shape)
    return matrix


def transform_texts(vectorizer: TfidfVectorizer, texts: Iterable[str]) -> spmatrix:
    """Transform texts using fitted vectorizer."""
    return vectorizer.transform(texts)
