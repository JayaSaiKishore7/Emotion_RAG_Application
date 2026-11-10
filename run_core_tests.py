"""Smoke test for core.models: python run_core_tests.py (exits non-zero on failure)."""

import io

from PIL import Image

from core import models

labels = models.load()["emotion_model"].config.id2label.values()

buffer = io.BytesIO()
Image.new("RGB", (224, 224), color="white").save(buffer, format="PNG")
emotion = models.detect_emotion(buffer.getvalue())
print("detect_emotion:", emotion)
assert emotion in labels

# A JPEG carrying an EXIF rotation (as phones produce) must be accepted.
exif = Image.Exif()
exif[0x0112] = 6  # Orientation: rotate 90° clockwise
buffer = io.BytesIO()
Image.new("RGB", (300, 200), color="gray").save(buffer, format="JPEG", exif=exif)
assert models.detect_emotion(buffer.getvalue()) in labels

# Real photo: the laughing man from the README screenshot. Classified whole, the model
# said "neutral"; cropped to the face it must say "happiness".
shot = Image.open("Deliverables (Stage 3)/happy.png").convert("RGB")
s = shot.width / 2000  # screenshot coordinates below are at 2000px width
buffer = io.BytesIO()
shot.crop((int(226 * s), int(449 * s), int(1074 * s), int(956 * s))).save(buffer, format="JPEG", quality=95)
emotion = models.detect_emotion(buffer.getvalue())
print("happy photo:", emotion)
assert emotion == "happiness", emotion

# Photo flow: every review must carry the detected emotion and be distinct.
for label in labels:
    result = models.search_reviews(f"What are customers saying who feel {label}?", label)
    texts = [r["text"] for r in result["reviews"]]
    print(f"{label}: {len(texts)} reviews | {result['summary'][:100]}")
    assert all(r["emotion"] == label for r in result["reviews"]), f"{label}: wrong emotion tag"
    assert len(texts) == len(set(texts)), f"{label}: duplicate reviews"
    assert len(texts) == 4 or label == "disgust", f"{label}: expected 4 reviews"

# Manual flow: one unnegated emotion word filters by that tag; anything else doesn't.
for query, expected in {
    "What makes customers sad?": "sadness",
    "Show me angry customer reviews": "anger",
    "What are customers happy about?": "happiness",
    "Why are people unhappy?": "sadness",
    "What scared buyers?": "fear",
    "What happened with deliveries?": None,  # "happen" is not "happy"
    "Who made a complaint?": None,  # "made" is not "mad"
    "Which customers are not happy?": None,  # negated
    "Happy or sad customers?": None,  # ambiguous
}.items():
    result = models.search_reviews(query)
    texts = [r["text"] for r in result["reviews"]]
    print(f"{query!r} -> {result['emotion']}")
    assert result["emotion"] == expected, f"{query!r}: got {result['emotion']}, expected {expected}"
    assert result["summary"] and len(texts) == 4 == len(set(texts)), query
    if expected:
        assert all(r["emotion"] == expected for r in result["reviews"]), query

sentiment = models.sentiment("This is a wonderful product!")
print("sentiment:", sentiment)
assert sentiment.startswith("POSITIVE")
assert models.sentiment("great product " * 600)  # longer than the model's 512-token limit

print("ALL CHECKS PASSED")
