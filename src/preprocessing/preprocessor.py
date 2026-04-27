import re
import unicodedata
from typing import List

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")  # capture word
_EMOJI_RE = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "]+",
    flags=re.UNICODE,
)
_PUNCT_RE = re.compile(r"[^\w\s!?]")
_EXTRA_SPACE_RE = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def replace_emojis(text: str) -> str:
    # Replace emojis with token instead of removing
    return _EMOJI_RE.sub(" EMOJI ", text)


def clean_text(text: str) -> str:
    """
    Optimized cleaning for sentiment classification.
    """
    if not isinstance(text, str):
        text = str(text)

    text = normalize_unicode(text)
    text = text.lower()

    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)

    # keep hashtag word
    text = _HASHTAG_RE.sub(r"\1", text)

    text = replace_emojis(text)

    # keep ! and ? for emotional intensity
    text = _PUNCT_RE.sub(" ", text)

    text = _EXTRA_SPACE_RE.sub(" ", text).strip()

    return text


def preprocess_texts(texts: List[str]) -> List[str]:
    return [clean_text(t) for t in texts]