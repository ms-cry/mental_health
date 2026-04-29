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

st.title("🧠 Mental Health Sentiment Analyzer")


# ── Helpers ───────────────────────────────────────────────

def truncate_text(text: str, max_chars: int = 150) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "..."


LABEL_EMOJI = {
    "normal": "✅",
    "sad": "💙",
    "anxiety": "😟",
    "depressive indicators": "🔴",
}


def show_results(results):
    """
    Display prediction results in Streamlit.
    """

    # Summary
    st.subheader("📊 Summary")

    counts = {}
    for r in results:
        counts[r["label"]] = counts.get(r["label"], 0) + 1

    cols = st.columns(4)
    labels = ["normal", "sad", "anxiety", "depressive indicators"]

    for col, label in zip(cols, labels):
        count = counts.get(label, 0)
        pct = (count / len(results)) * 100 if results else 0

        col.metric(
            label=f"{LABEL_EMOJI[label]} {label}",
            value=count,
            delta=f"{pct:.1f}%"
        )

    # Detailed Results
    st.subheader("📋 Comment Analysis")

    for r in results:

        if r["label"] == "depressive indicators":
            st.error(
                f"⚠️ HIGH RISK SIGNAL\n\n"
                f"{truncate_text(r['text'])}"
            )
        else:
            st.markdown(
                f"**{truncate_text(r['text'])}**"
            )

        st.caption(r["interpretation"])
        st.write(
            f"{LABEL_EMOJI[r['label']]} **{r['label']}**"
        )
        st.divider()


# ── Input Section ─────────────────────────────────────────

st.subheader("🎥 YouTube Video Analysis")

video_url = st.text_input(
    "Enter YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

max_comments = st.slider(
    "Number of comments to analyze",
    min_value=20,
    max_value=200,
    value=100,
    step=10,
)

st.markdown("---")

st.subheader("✍️ Manual Comment Input")

manual_input = st.text_area(
    "Enter comments manually (one comment per line)",
    height=200,
    placeholder="""
I feel tired and alone
I am stressed about exams
Life is going well today
Nothing matters anymore
"""
)

run = st.button(
    "Analyze Comments",
    use_container_width=True
)


# ── Main Pipeline ─────────────────────────────────────────

if run:

    comments = []

    # Priority 1 → Manual Input
    if manual_input.strip():
        comments = [
            line.strip()
            for line in manual_input.split("\n")
            if line.strip()
        ]

    # Priority 2 → YouTube URL
    elif video_url.strip():
        fetcher = YouTubeFetcher(
            max_comments=max_comments
        )

        with st.spinner("Fetching YouTube comments..."):
            comments = fetcher.fetch_comments(video_url)

    else:
        st.error(
            "Please enter either:\n"
            "- YouTube URL\n"
            "- Manual comments"
        )
        st.stop()

    if not comments:
        st.warning(
            "No valid comments found."
        )
        st.stop()

    # Prediction
    with st.spinner("Analyzing sentiment..."):
        try:
            results = predict(comments)

        except FileNotFoundError:
            st.error(
                "Model not found.\n\n"
                "Run training first:\n\n"
                "python -m src.models.train"
            )
            st.stop()

        except Exception as e:
            st.error(
                f"Prediction error: {str(e)}"
            )
            st.stop()

    show_results(results)


# ── Demo Mode ─────────────────────────────────────────────

with st.expander("Try Demo (No YouTube needed)"):

    if st.button("Run Demo"):

        demo_comments = [
            "I feel empty and tired",
            "I am stressed about exams",
            "Life is amazing today",
            "Nothing matters anymore",
            "I feel nervous about tomorrow",
        ]

        try:
            results = predict(demo_comments)
            show_results(results)

        except FileNotFoundError:
            st.warning(
                "Train the model first:\n\n"
                "python -m src.models.train"
            )