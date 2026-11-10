"""Emotion detection, review retrieval, summarization and sentiment.

The app talks to this module through four functions: `load`, `detect_emotion`,
`search_reviews` and `sentiment`. Models load once per process (cache_resource);
results are cached per input (cache_data), so Streamlit reruns stay cheap.
"""

import io
import re

import cv2
import numpy as np
import streamlit as st
import torch
from huggingface_hub import snapshot_download
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

EMOTION_MODEL_REPO = "primal-sage/emotion-model"
RAG_INDEX_REPO = "primal-sage/my-rag-index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Exact commits to load. Without these, anyone who can push to a repo changes what
# runs here on the next restart; that includes the index, a pickle that executes code
# when loaded. Embeddings must also match the ones the index was built with.
# To upgrade: test against the new commit, then update the hash here.
REVISIONS = {
    EMOTION_MODEL_REPO: "b6db94c4536e2ed824906df031d2d9225cd541bc",
    RAG_INDEX_REPO: "641233bc2618cafd435a01c2580ada0cd6e40d55",
    EMBEDDING_MODEL: "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    LLM_MODEL: "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    SENTIMENT_MODEL: "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7",
}

# Emotion words in a typed query -> review tag to filter on. Patterns anchor at the
# start of a word, so "unhappy" doesn't count as happiness and "made" isn't "mad".
QUERY_EMOTIONS = {
    "anger": r"\b(anger|angr|mad\b|furious|rage|annoy|irritat|frustrat)",
    "disgust": r"\b(disgust|gross|revolt|repuls|nause)",
    "fear": r"\b(fear|afraid|scare|scary|frighten|worr|anxi|nervous|alarm)",
    "happiness": r"\b(happy|happi|joy|delight|pleased|glad|thrill|love)",  # not "happen"
    "sadness": r"\b(sad|unhapp|sorrow|heartbr|disappoint|depress|upset|melanchol|regret)",
    "surprise": r"\b(surpris|shock|amaz|astonish|unexpect|startl)",
    "neutral": r"\b(neutral|indifferent|average|ordinary|okay|ok\b|meh\b)",
}
# "not happy", "wasn't angry": the word alone would pick the wrong tag, so don't filter.
NEGATION = r"\b(not|no|never|without|hardly)\b|n't\b"

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

    emotion_model = AutoModelForImageClassification.from_pretrained(
        EMOTION_MODEL_REPO, revision=REVISIONS[EMOTION_MODEL_REPO]
    ).to(device).eval()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device, "revision": REVISIONS[EMBEDDING_MODEL]},
        encode_kwargs={"normalize_embeddings": True},
    )
    # Downloads once into the Hugging Face cache; later starts reuse it.
    index_dir = snapshot_download(
        RAG_INDEX_REPO, revision=REVISIONS[RAG_INDEX_REPO], allow_patterns=["index.faiss", "index.pkl"]
    )

    return {
        "device": device,
        "emotion_processor": AutoImageProcessor.from_pretrained(EMOTION_MODEL_REPO, revision=REVISIONS[EMOTION_MODEL_REPO]),
        "emotion_model": emotion_model,
        # Safe only because the commit is pinned above; FAISS requires opting in to load it.
        "db": FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True),
        "llm": HuggingFacePipeline(
            pipeline=pipeline(
                "text2text-generation", model=LLM_MODEL, revision=REVISIONS[LLM_MODEL], max_length=150, device=device_id
            )
        ),
        "sentiment": pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            revision=REVISIONS[SENTIMENT_MODEL],
            device=device_id,
            truncation=True,
            max_length=512,  # the model sets no limit itself; longer reviews would crash
        ),
    }


# OpenCV's bundled frontal-face detector. OpenCV 5 dropped it, hence the <5 pin.
FACE_DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@st.cache_data(show_spinner=False, max_entries=256)
def detect_emotion(image_bytes: bytes) -> str:
    """Return the predicted emotion label (e.g. "happiness") for an image.

    The model was trained on tight face crops (FER-2013). On 350 FER test faces
    placed inside larger photos, classifying the whole photo scored 20%; cropping
    to the face first scored 44%, close to the 46% on the original close-ups.
    The crop is tight (no padding): extra margin flipped a laughing face to "anger".
    Without a detected face, the whole photo is classified as before.
    """
    m = load()
    # Phone photos store rotation in EXIF; without this the model sees a sideways face.
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    faces = FACE_DETECTOR.detectMultiScale(np.array(image.convert("L")), scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces):
        # Largest detection: background texture can produce small false faces.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        image = image.crop((x, y, x + w, y + h))
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

    `emotion` restricts results to reviews tagged with that emotion. When it's None,
    a single unnegated emotion word in the query ("angry", "sad") sets it. Returns
    {"summary": str, "emotion": str | None, "reviews": [{"text", "emotion", "sentiment"}, ...]}.
    """
    if emotion is None and not re.search(NEGATION, query, re.IGNORECASE):
        found = [label for label, pattern in QUERY_EMOTIONS.items() if re.search(pattern, query, re.IGNORECASE)]
        emotion = found[0] if len(found) == 1 else None
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
        return {"summary": "No relevant documents found for your query.", "emotion": emotion, "reviews": []}
    context = "\n\n".join(doc.page_content for doc in docs)
    return {
        "summary": m["llm"].invoke(PROMPT_TEMPLATE.format(context=context, question=query)),
        "emotion": emotion,
        "reviews": [
            {
                "text": doc.page_content,
                "emotion": doc.metadata.get("emotion", "Unknown"),
                "sentiment": sentiment(doc.page_content),
            }
            for doc in docs
        ],
    }
