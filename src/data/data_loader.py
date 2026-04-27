from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.config import LABEL_MAP, RANDOM_STATE

logger = logging.getLogger(__name__)

def load_dataset(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)

    print("Original rows:", len(df))

    # Clean text
    df["text"] = df["text"].astype(str).str.strip()

    # 🔥 FIX: Map BEFORE numeric conversion
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    label_mapping = {
        "normal": 0,

        "stress": 1,

        "anxiety": 2,

        "depression": 3,
        "suicidal": 3,
        "bipolar": 3,
        "personality disorder": 3
    }

    df["label"] = df["label"].map(label_mapping)

    print("After mapping:")
    print(df["label"].value_counts(dropna=False))

    # Now convert numeric
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    # Drop bad rows
    df.dropna(subset=["text", "label"], inplace=True)
    df = df[df["text"] != ""]
    df["label"] = df["label"].astype(int)

    print("Final rows:", len(df))
    print("Final label distribution:")
    print(df["label"].value_counts())

    return df

def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
):
    """
    Simple train-test split (aligned with train.py)
    """

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    logger.info(
        "Split → train=%d  test=%d",
        len(train_df),
        len(test_df),
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_sample_dataset() -> pd.DataFrame:
    """
    Small built-in dataset for testing.
    """

    samples = [
        ("I had a great day today!", 0),
        ("Feeling happy and productive.", 0),

        ("I feel very sad and alone.", 1),
        ("Everything feels pointless.", 1),

        ("I am constantly worried about everything.", 2),
        ("My anxiety is getting worse.", 2),

        ("Nothing matters anymore.", 3),
        ("I feel completely hopeless.", 3),
    ]

    df = pd.DataFrame(samples, columns=["text", "label"])

    logger.info("Loaded sample dataset (%d rows)", len(df))

    return df