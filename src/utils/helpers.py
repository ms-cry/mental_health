import re
from typing import Optional

from src.config.config import LABEL_MAP, INTERPRETATIONS


def get_label_name(label_id: int) -> str:
    """
    Convert numeric label to text label.
    """
    return LABEL_MAP.get(label_id, "normal")


def get_interpretation(label_id: int) -> str:
    """
    Return safe interpretation based on label ID.
    """
    label = get_label_name(label_id)
    return INTERPRETATIONS.get(
        label,
        "Unable to determine emotional signal from this comment."
    )


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from URL.
    """
    patterns = [
        r"(?:v=)([0-9A-Za-z_-]{11})",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"embed/([0-9A-Za-z_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None