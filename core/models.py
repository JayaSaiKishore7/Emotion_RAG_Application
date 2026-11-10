"""Emotion detection, review retrieval, summarization and sentiment.

The app talks to this module through four functions: `load`, `detect_emotion`,
`search_reviews` and `sentiment`. Models load once per process (cache_resource);
results are cached per input (cache_data), so Streamlit reruns stay cheap.
"""

import io

import streamlit as st
import torch
from huggingface_hub import snapshot_download
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

EMOTION_MODEL_REPO = "primal-sage/emotion-model"
RAG_INDEX_REPO = "primal-sage/my-rag-index"

PROMPT_TEMPLATE = """Based on the customer reviews below, provide a clear and concise summary that directly answers the question.

Customer Reviews:
{context}

Question: {question}

Provide a brief summary (2-3 sentences maximum):"""


@st.cache_resource(show_spinner=False)
def load():
    """Load every model once per process and return them as a dict."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1

    emotion_model = AutoModelForImageClassification.from_pretrained(EMOTION_MODEL_REPO).to(device).eval()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    # Downloads once into the Hugging Face cache; later starts reuse it.
    index_dir = snapshot_download(RAG_INDEX_REPO, allow_patterns=["index.faiss", "index.pkl"])

    return {
        "device": device,
        "emotion_processor": AutoImageProcessor.from_pretrained(EMOTION_MODEL_REPO),
        "emotion_model": emotion_model,
        # The pickle comes from our own index repo; FAISS requires opting in to load it.
        "db": FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True),
        "llm": HuggingFacePipeline(
            pipeline=pipeline("text2text-generation", model="google/flan-t5-base", max_length=150, device=device_id)
        ),
        "sentiment": pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device_id,
            truncation=True,
            max_length=512,  # the model sets no limit itself; longer reviews would crash
        ),
    }


@st.cache_data(show_spinner=False, max_entries=256)
def detect_emotion(image_bytes: bytes) -> str:
    """Return the predicted emotion label (e.g. "happiness") for an image."""
    m = load()
    # Phone photos store rotation in EXIF; without this the model sees a sideways face.
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    inputs = m["emotion_processor"](images=image, return_tensors="pt").to(m["device"])
    with torch.no_grad():
        logits = m["emotion_model"](**inputs).logits
    return m["emotion_model"].config.id2label[logits.argmax(-1).item()]


@st.cache_data(show_spinner=False)
def sentiment(text: str) -> str:
    """Return e.g. "POSITIVE (Score: 0.98)"."""
    result = load()["sentiment"](text)[0]
    return f"{result['label'].upper()} (Score: {result['score']:.2f})"


@st.cache_data(show_spinner=False, max_entries=256)
def search_reviews(query: str, emotion: str | None = None) -> dict:
    """Retrieve the 4 most relevant distinct reviews for `query` and summarize them.

    `emotion` restricts results to reviews tagged with that emotion. Returns
    {"summary": str, "reviews": [{"text", "emotion", "sentiment"}, ...]}.
    """
    m = load()
    # The index holds 500 reviews but only 60 distinct texts, so a plain top-4 is
    # usually one review four times, and the query wording alone sends "happiness"
    # to sad reviews. Rank the whole (small) index, filter by tag, keep first copies.
    total = m["db"].index.ntotal
    hits = m["db"].similarity_search(query, k=total, fetch_k=total, filter={"emotion": emotion} if emotion else None)
    unique = {}
    for doc in hits:
        unique.setdefault(doc.page_content, doc)
    docs = list(unique.values())[:4]
    if not docs:
        return {"summary": "No relevant documents found for your query.", "reviews": []}
    context = "\n\n".join(doc.page_content for doc in docs)
    return {
        "summary": m["llm"].invoke(PROMPT_TEMPLATE.format(context=context, question=query)),
        "reviews": [
            {
                "text": doc.page_content,
                "emotion": doc.metadata.get("emotion", "Unknown"),
                "sentiment": sentiment(doc.page_content),
            }
            for doc in docs
        ],
    }
