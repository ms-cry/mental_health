from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.data.youtube_fetcher import YouTubeFetcher
from src.models.predict import predict


# ── UI Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Mental Health Sentiment Analyzer (YouTube Comments)")


# ── Helper ───────────────────────────────────────────────
def truncate_text(text: str, max_chars: int = 150) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "..."


LABEL_EMOJI = {
    "normal": "✅",
    "sad": "💙",
    "anxiety": "😟",
    "depressive indicators": "🔴",
}


# ── Input Section ─────────────────────────────────────────
video_url = st.text_input(
    "Enter YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

max_comments = st.slider(
    "Number of comments to analyze",
    20, 200, 100, step=10
)

run = st.button("Analyze Comments", use_container_width=True)


# ── Pipeline ─────────────────────────────────────────────
if run:

    if not video_url.strip():
        st.error("Please enter a valid YouTube URL")
        st.stop()

    fetcher = YouTubeFetcher(max_comments=max_comments)

    # Fetch comments
    with st.spinner("Fetching comments..."):
        comments = fetcher.fetch_comments(video_url)

    if not comments:
        st.warning("No comments found or comments disabled")
        st.stop()

    # Predict
    with st.spinner("Analyzing sentiment..."):
        try:
            results = predict(comments)
        except FileNotFoundError:
            st.error("Model not found. Run training first:\n\npython -m src.models.train")
            st.stop()
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    # ── Summary ─────────────────────────────────────────
    st.subheader("📊 Summary")

    counts = {}
    for r in results:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    cols = st.columns(4)
    labels = ["normal", "sad", "anxiety", "depressive indicators"]

    for col, label in zip(cols, labels):
        count = counts.get(label, 0)
        pct = (count / len(results)) * 100

        col.metric(
            label=f"{LABEL_EMOJI[label]} {label}",
            value=count,
            delta=f"{pct:.1f}%"
        )

    # ── Results ─────────────────────────────────────────
    st.subheader("📋 Comment Analysis")

    for r in results:

        # Highlight risky comments
        if r["label"] == "depressive indicators":
            st.error(f"⚠️ HIGH RISK\n\n{truncate_text(r['text'])}")
        else:
            st.markdown(f"**{truncate_text(r['text'])}**")

        st.caption(r["interpretation"])
        st.write(f"{LABEL_EMOJI[r['label']]} **{r['label']}**")
        st.divider()


# ── Demo Mode ───────────────────────────────────────────
with st.expander("Try Demo (No YouTube needed)"):

    if st.button("Run Demo"):

        demo = [
            "I feel empty and tired",
            "I am stressed about exams",
            "Life is amazing today",
            "Nothing matters anymore",
        ]

        results = predict(demo)

        for r in results:
            st.markdown(f"- {LABEL_EMOJI[r['label']]} {truncate_text(r['text'])}")
            st.caption(r["interpretation"])