# Facial Emotion-Driven Review Analysis System

**Live Demo:** [https://emotionragapplication-dmj76csbefwfku6hgnchkt.streamlit.app/](https://emotionragapplication-dmj76csbefwfku6hgnchkt.streamlit.app/)

---

## Overview

This project detects facial emotions from images and uses that emotion to retrieve and summarize relevant customer reviews. Built for the Junior AI Engineer assignment.

**Example:** Upload a "happy" face → System finds reviews explaining *why* customers felt happy.

### Key Features

* **Emotion Detection**: Classifies 7 emotions (Happy, Sad, Angry, Disgust, Fear, Surprise, Neutral)
* **RAG System**: Links emotions to relevant reviews using FAISS vector search
* **Sentiment Analysis**: Analyzes sentiment of each review
* **Summary**: One line counting the retrieved reviews' sentiment (e.g. "3 of 4 reviews are negative, 1 is neutral.")
* **Web Interface**: Streamlit app for easy interaction

### Tech Stack

* **Vision Model**: ResNet-50 (fine-tuned on FER-2013), applied to the largest detected face
* **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
* **Vector Store**: FAISS
* **Sentiment**: cardiffnlp/twitter-roberta-base-sentiment-latest
* **Framework**: LangChain
* **UI**: Streamlit

---

## Architecture

```text
User uploads image
         ↓
Face Crop (OpenCV face detector)
         ↓
Emotion Detection (ResNet-50)
         ↓
Query Generation
         ↓
RAG Pipeline (FAISS + LangChain)
         ↓
Sentiment Analysis (RoBERTa) + one-line sentiment summary
         ↓
Display Results
```

---

## Stage 1: Emotion Recognition

**Model:** Fine-tuned ResNet-50 on FER-2013 dataset (28,709 training images, 7 emotion classes)

**Results** (FER-2013 test set, 3,589 images):
* Test Accuracy: **62.2%**
* Weighted F1-Score: **0.61** (macro F1: 0.52)
* Best performing: Happy (89.1%)
* Most challenging: Disgust (3.6%, 2 of 55; disgust is the rarest class and is usually predicted as anger), then Fear (32.6%)

### Confusion Matrix

![Confusion Matrix](Deliverables%20(Stage%201)/confusion_matrix.png)

### Per-Class Accuracy

![Per-Class Accuracy](Deliverables%20(Stage%201)/pre_Class%20acuuracy.png)

---

## Stage 2: RAG & Sentiment Pipeline

### Data
Generated 500 synthetic reviews to simulate customer feedback, each built from a Stage 1 prediction and labelled with its predicted emotion: happiness 128, neutral 120, sadness 87, anger 58, surprise 54, fear 53. Disgust has no reviews because the model almost never predicts it.

### RAG Pipeline Components

1. **Embeddings**: Converted reviews to vectors using `all-MiniLM-L6-v2`
2. **Vector Store**: Stored in FAISS index for fast similarity search
3. **Retrieval**: LangChain retrieves top-4 relevant reviews
4. **Sentiment**: RoBERTa classifies each review sentiment
5. **Summary**: a one-line count of those sentiments. The original Flan-T5 summarizer was removed: `flan-t5-base` only copied reviews verbatim, under every prompt tested

---

## Stage 3: Design Decisions

### 1. Technology Choices

* **ResNet-50**: Proven CNN architecture, faster training than ViT for small datasets
* **FAISS**: Efficient local vector search, perfect for 500 reviews
* **LangChain**: Simplified RAG pipeline development
* **No LLM summarizer**: Flan-T5 was tried first; at `base` size it copied reviews instead of summarizing, and larger models don't fit Streamlit Cloud's memory. A sentiment count is honest and free

### 2. Data Flow

`Image (bytes)` → `Emotion (string)` → `Query (string)` → `Embeddings (vector)` → `FAISS Search` → `Reviews (text)` → `Sentiment + Summary line` → `UI Display`

**Deployment:** Would use FastAPI with 3 endpoints:
- `POST /predict_emotion` - Image → Emotion
- `POST /query_rag` - Query → Summary + Reviews
- `POST /analyze_sentiment` - Text → Sentiment

### 3. Scalability

For millions of reviews:
* **Vector DB**: Switch to Pinecone or Weaviate (managed, distributed)
* **Embeddings**: Upgrade to `all-mpnet-base-v2` or API-based models
* **Indexing**: Use HNSW index for faster search

### 4. Ethics & Bias

* **Vision Bias**: FER-2013 has demographic imbalances. Solution: Retrain on diverse datasets (FairFace, RAF-DB)
* **NLP Bias**: Synthetic reviews may reinforce stereotypes. Solution: Use real, diverse review data + user feedback system

---

## App Screenshots

### Happy Emotion
![Happy](Deliverables%20(Stage%203)/happy.png)

### Angry Emotion
![Angry](Deliverables%20(Stage%203)/angry.png)

### Sad Emotion
![Sad](Deliverables%20(Stage%203)/sad.png)

### Surprised Emotion
![Surprised](Deliverables%20(Stage%203)/surprise.png)

---

## Project Structure

```text
Emotion_RAG_Application/
├── Deliverables (Stage 1)/
│   ├── model/                    # Fine-tuned ResNet-50
│   ├── confusion_matrix.png
│   ├── pre_Class acuuracy.png
│   ├── evaluation_metrics.txt
│   ├── training_metrics.txt
│   └── stage1_predictions.csv
├── Deliverables (Stage 2)/
│   ├── rag/                      # FAISS index files
│   ├── generated_reviews.csv
│   ├── query_results.csv
│   └── stage2_report.txt
├── Deliverables (Stage 3)/       # App screenshots
├── .devcontainer/               # Dev container config
├── AI_part1.ipynb               # Stage 1: Model training
├── AI_part2.ipynb               # Stage 2: RAG pipeline
├── core/models.py               # Model loading, emotion, retrieval, sentiment
├── app.py                       # Streamlit web interface
├── run_core_tests.py            # Smoke test for core/models.py
├── .github/workflows/ci.yml     # Lint + smoke test; promotes passing main to prod
├── requirements.in              # Direct dependencies (edit this)
├── requirements.txt             # Full lock generated from requirements.in
└── README.md
```

---

## How to Run

### Prerequisites
- Python 3.11–3.13 (3.12 recommended; on Streamlit Cloud pick 3.12 under Advanced settings, since 3.14 can't install `pyarrow` for Streamlit 1.51)
- Git

### Installation

```bash
git clone https://github.com/JayaSaiKishore7/Emotion_RAG_Application.git
cd Emotion_RAG_Application

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py

# Optional smoke test of the model pipeline
python run_core_tests.py
```

App opens at `http://localhost:8501`

---

## Deployment

The live app runs on Streamlit Community Cloud from this repo's **`prod`** branch (`app.py`). Nobody pushes to `prod` by hand:

```text
push to main → CI: ruff + run_core_tests.py → passed? → CI fast-forwards prod → Streamlit redeploys
                                            → failed? → prod stays on the last good commit
```

### Before pushing
- Locally: `python run_core_tests.py` must print `ALL CHECKS PASSED`.
- After pushing, check the commit's CI result on GitHub (Actions tab). A red result means it did not go live.
- If `test` passes but `promote` fails, `prod` didn't move. Run `git push origin main:prod` yourself.

### When CI fails without running
- **"The job was not started because your account is locked due to a billing issue":** no code ran. Fix billing at github.com/settings/billing, then re-run the run.
- **Re-run only the newest run on `main`.** CI cancels a running build when another one starts on the same branch, so re-running several old runs at once cancels the newest commit's run ("Canceling since a higher priority waiting request ... exists").

### Dependencies
`requirements.in` lists the direct dependencies; `requirements.txt` is the full lock of all ~120 packages, which is what Streamlit and CI install. To add or upgrade a package, edit `requirements.in` and regenerate the lock with the command at the top of `requirements.txt` (needs `pip install uv`). A loose lock once broke a rebuild with no code change: `pyarrow` stopped installing.

### Settings that live outside the repo
- **Deploy branch `prod`**, file `app.py`, chosen when creating the app. The branch can't be changed afterwards: to change it, delete the app and recreate it. Keep the new auto-generated URL and update the Live Demo link above: an app recreated right away on the deleted app's URL got stuck loading with empty logs.
- **Python 3.12**, under Advanced settings when creating the app, or later under the app's Settings. The default (3.14) can't install `pyarrow` for Streamlit 1.51.
- **No environment variables or secrets are needed.** All Hugging Face models are public and pinned by commit in `core/models.py`.
- **Don't use Git LFS.** The GitHub LFS quota ran out once and clones failed; the model and images are regular Git files.

### Changing a model or the review index
`REVISIONS` in `core/models.py` pins each Hugging Face repo to a commit. The review index (`primal-sage/my-rag-index`) was built with `all-MiniLM-L6-v2` embeddings, so if either one changes, change both together. Run `run_core_tests.py` before pushing: search quietly returns wrong reviews if they don't match.

### After deploying
Open the live app, upload a face photo, and check that an emotion and four reviews appear. The URL returning a page doesn't prove the app works, because Streamlit shows its page even when the app crashed. Errors are in the app's logs ("Manage app" in the bottom-right corner, visible to the owner only).

### Rolling back
1. Fastest: point `prod` back at the last good commit with `git push --force origin <good commit>:prod`. Streamlit redeploys it; `main` is untouched. Find the commit with `git log origin/prod`.
2. Then `git revert <bad commit>` and push to `main`. Do this before pushing anything else: the next push to `main` that passes CI moves `prod` forward again, bad commit included unless it was reverted.
3. If the app is stuck or still failing, open the Streamlit dashboard and choose **Reboot app**.

