"""Streamlit app for Facial Emotion-Driven Review Analysis."""

import io

import streamlit as st
from PIL import Image

from core import models

# Model label -> (adjective used in the review query, emoji shown with the result)
EMOTIONS = {
    "anger": ("angry", "😠"),
    "disgust": ("disgusted", "🤢"),
    "fear": ("afraid", "😨"),
    "happiness": ("happy", "😄"),
    "sadness": ("sad", "😢"),
    "surprise": ("surprised", "😲"),
    "neutral": ("neutral", "😐"),
}

st.set_page_config(page_title="Facial Emotion RAG System", page_icon="🤖", layout="wide")

st.title("Facial Emotion-Driven Review Analysis 🤖")
st.write("Upload a photo of a face. The app detects the emotion, then finds and summarizes customer reviews written in that mood.")

with st.spinner("🚀 Warming up the AI models... This may take a moment the first time."):
    models.load()


def show_results(result: dict, empty_message: str):
    st.subheader("🔍 AI Summary")
    st.write(result["summary"])
    if not result["reviews"]:
        st.warning(empty_message)
        return
    st.subheader("📚 Retrieved Reviews")
    for i, review in enumerate(result["reviews"], 1):
        with st.container(border=True):
            col_a, col_b = st.columns([3, 1])
            col_a.markdown(f"**Review #{i}**")
            col_a.markdown(f"*Emotion Tag: {review['emotion']}*")
            col_b.markdown("**Sentiment**")
            col_b.markdown(review["sentiment"])
            st.write(review["text"])


tab1, tab2 = st.tabs(["📸 Query with Photo", "💬 Manual Query"])

with tab1:
    st.header("Upload a Photo → Detect Emotion → Query Reviews")
    col1, col2 = st.columns([1, 1])
    uploaded_file = col1.file_uploader(
        "Choose a photo of a face...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of a face for emotion detection",
    )
    image_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
    if image_bytes is not None:
        # Validate once at the boundary: a renamed or corrupt file must not crash the page.
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except Exception:
            col2.error("This file can't be read as an image. Please upload a JPG or PNG photo.")
            image_bytes = None
    if image_bytes is not None:
        col1.image(uploaded_file, caption="Uploaded Image", width="stretch")
        with col2:
            with st.spinner("🔍 Analyzing facial emotion..."):
                emotion = models.detect_emotion(image_bytes)
            adjective, emoji = EMOTIONS[emotion]
            st.success(f"**Detected Emotion:** {emotion.upper()} {emoji}")
            dynamic_query = f"What are customers saying who feel {adjective}?"
            st.info(f"**Dynamic Query:** *'{dynamic_query}'*")
            with st.spinner(f"📚 Searching review database for '{emotion}' reviews..."):
                result = models.search_reviews(dynamic_query, emotion)
            show_results(result, "No reviews found for this emotion.")

with tab2:
    st.header("Query the Review Database Manually")
    st.text_input(
        "Enter your query:",
        key="query",
        placeholder="e.g., 'What are customers happy about?'",
        help="Ask any question about customer reviews",
    )
    st.markdown("**Example queries:**")
    examples = ["What are customers happy about?", "Show me angry customer reviews", "What makes customers sad?"]
    # Example buttons fill the box and search in one click. on_click runs before the
    # script reruns, so the text input can still be updated through session_state.
    for col, example in zip(st.columns(3), examples):
        col.button(
            f"📝 {example}",
            key=f"example_{example}",
            on_click=st.session_state.update,
            kwargs={"query": example, "searched": example},
        )
    if st.button("🔍 Search Reviews", type="primary"):
        st.session_state.searched = st.session_state.query
        if not st.session_state.query:
            st.warning("Please enter a query.")
    # Results stay up for the last searched query through any later rerun.
    if st.session_state.get("searched"):
        with st.spinner("Searching review database..."):
            result = models.search_reviews(st.session_state.searched)
        if result["emotion"]:
            adjective, emoji = EMOTIONS[result["emotion"]]
            st.caption(f"{emoji} Your question mentions feeling {adjective}, so only reviews tagged *{result['emotion']}* are shown.")
        show_results(result, "No reviews found matching your query.")
