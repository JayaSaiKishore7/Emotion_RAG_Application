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

# Photo flow: every review must carry the detected emotion and be distinct.
for label in labels:
    result = models.search_reviews(f"What are customers saying who feel {label}?", label)
    texts = [r["text"] for r in result["reviews"]]
    print(f"{label}: {len(texts)} reviews | {result['summary'][:100]}")
    assert all(r["emotion"] == label for r in result["reviews"]), f"{label}: wrong emotion tag"
    assert len(texts) == len(set(texts)), f"{label}: duplicate reviews"
    assert len(texts) == 4 or label == "disgust", f"{label}: expected 4 reviews"

# Manual flow: distinct reviews, no emotion filter.
result = models.search_reviews("What are customers happy about?")
texts = [r["text"] for r in result["reviews"]]
assert result["summary"] and len(texts) == 4 == len(set(texts))

sentiment = models.sentiment("This is a wonderful product!")
print("sentiment:", sentiment)
assert sentiment.startswith("POSITIVE")
assert models.sentiment("great product " * 600)  # longer than the model's 512-token limit

print("ALL CHECKS PASSED")
