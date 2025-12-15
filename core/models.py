# model_utils.py - COMPLETE WORKING VERSION WITH FULL DEBUG
"""
This is a complete working version that avoids the deprecated RetrievalQA
and uses a simplified approach that will work with current langchain versions.
"""

import streamlit as st
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    pipeline
)
from PIL import Image
import numpy as np
import os
from huggingface_hub import hf_hub_download
import time
import sys
import traceback

# Debug: Print Python and package versions
print(f"DEBUG: Python version: {sys.version}")
print(f"DEBUG: Current working directory: {os.getcwd()}")

# Try different import strategies for langchain components
print("DEBUG: Starting langchain imports...")

# Import 1: Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("DEBUG: ✓ Successfully imported HuggingFaceEmbeddings from langchain_huggingface")
except ImportError as e:
    print(f"DEBUG: Failed to import from langchain_huggingface: {e}")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("DEBUG: ✓ Successfully imported HuggingFaceEmbeddings from langchain_community.embeddings")
    except ImportError as e2:
        print(f"DEBUG: Failed to import from langchain_community.embeddings: {e2}")
        from langchain.embeddings import HuggingFaceEmbeddings
        print("DEBUG: ✓ Successfully imported HuggingFaceEmbeddings from langchain.embeddings")

# Import 2: Vector Store
try:
    from langchain_community.vectorstores import FAISS
    print("DEBUG: ✓ Successfully imported FAISS from langchain_community.vectorstores")
except ImportError as e:
    print(f"DEBUG: Failed to import FAISS: {e}")
    from langchain.vectorstores import FAISS
    print("DEBUG: ✓ Successfully imported FAISS from langchain.vectorstores")

# Import 3: LLM Pipeline
try:
    from langchain_community.llms import HuggingFacePipeline
    print("DEBUG: ✓ Successfully imported HuggingFacePipeline from langchain_community.llms")
except ImportError as e:
    print(f"DEBUG: Failed to import from langchain_community.llms: {e}")
    from langchain.llms import HuggingFacePipeline
    print("DEBUG: ✓ Successfully imported HuggingFacePipeline from langchain.llms")

# Import 4: Prompts
try:
    from langchain.prompts import PromptTemplate
    print("DEBUG: ✓ Successfully imported PromptTemplate")
except ImportError as e:
    print(f"DEBUG: Failed to import PromptTemplate: {e}")
    from langchain_core.prompts import PromptTemplate
    print("DEBUG: ✓ Successfully imported PromptTemplate from langchain_core")

# --- Your Hugging Face Repos ---
EMOTION_MODEL_REPO = "primal-sage/emotion-model" 
RAG_FILES_REPO = "primal-sage/my-rag-index" 

# --- Local directory to store downloaded index ---
FAISS_INDEX_DIR = "faiss_index_local"

print(f"DEBUG: Emotion Model Repo: {EMOTION_MODEL_REPO}")
print(f"DEBUG: RAG Files Repo: {RAG_FILES_REPO}")
print(f"DEBUG: FAISS Index Directory: {FAISS_INDEX_DIR}")
print("DEBUG: model_utils.py imports completed successfully!")

@st.cache_resource
def load_emotion_model():
    """Loads the fine-tuned emotion classification model and processor."""
    print("DEBUG: === load_emotion_model START ===")
    st.write("Loading emotion model... (This may take a moment)")
    
    try:
        print(f"DEBUG: Loading processor from: {EMOTION_MODEL_REPO}")
        processor = AutoImageProcessor.from_pretrained(EMOTION_MODEL_REPO)
        print("DEBUG: ✓ Processor loaded successfully")
    except Exception as e:
        print(f"DEBUG: ✗ Failed to load processor: {e}")
        st.error(f"Failed to load processor: {e}")
        raise
    
    try:
        print(f"DEBUG: Loading model from: {EMOTION_MODEL_REPO}")
        model = AutoModelForImageClassification.from_pretrained(EMOTION_MODEL_REPO)
        print("DEBUG: ✓ Model loaded successfully")
    except Exception as e:
        print(f"DEBUG: ✗ Failed to load model: {e}")
        st.error(f"Failed to load model: {e}")
        raise
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEBUG: Setting emotion model device to: {device}")
    
    try:
        model.to(device)
        print(f"DEBUG: ✓ Model moved to {device}")
    except Exception as e:
        print(f"DEBUG: ✗ Failed to move model to {device}: {e}")
        raise
    
    print("DEBUG: === load_emotion_model END ===")
    return processor, model, device

PROMPT_TEMPLATE = """Based on the customer reviews below, provide a clear and concise summary that directly answers the question.

Customer Reviews:
{context}

Question: {question}

Provide a brief summary (2-3 sentences maximum):"""

@st.cache_resource
def load_rag_pipeline():
    """Loads all components for the RAG and Sentiment pipelines - SIMPLIFIED VERSION."""
    print("DEBUG: === load_rag_pipeline START ===")
    st.write("Loading RAG & Sentiment pipelines...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1
    print(f"DEBUG: RAG device: {device} (device_id: {device_id})")

    # Load embeddings
    try:
        print("DEBUG: Loading HuggingFaceEmbeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("DEBUG: ✓ Embeddings loaded successfully")
    except Exception as e:
        print(f"DEBUG: ✗ Failed to load embeddings: {e}")
        st.error(f"Failed to load embeddings: {e}")
        raise

    # Download FAISS index files
    try:
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        print(f"DEBUG: Created/verified directory: {FAISS_INDEX_DIR}")
        
        print(f"DEBUG: Downloading index.faiss from {RAG_FILES_REPO}")
        faiss_path = hf_hub_download(
            repo_id=RAG_FILES_REPO, 
            filename="index.faiss", 
            local_dir=FAISS_INDEX_DIR, 
            force_download=True
        )
        print(f"DEBUG: ✓ Downloaded index.faiss to: {faiss_path}")
        
        print(f"DEBUG: Downloading index.pkl from {RAG_FILES_REPO}")
        pkl_path = hf_hub_download(
            repo_id=RAG_FILES_REPO, 
            filename="index.pkl", 
            local_dir=FAISS_INDEX_DIR, 
            force_download=True
        )
        print(f"DEBUG: ✓ Downloaded index.pkl to: {pkl_path}")
    except Exception as e:
        print(f"DEBUG: ✗ Failed to download RAG files: {e}")
        st.error(f"Failed to download RAG files: {e}")
        raise

    # Load FAISS index
    try:
        print(f"DEBUG: Loading FAISS index from {FAISS_INDEX_DIR}")
        db = FAISS.load_local(
            FAISS_INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print(f"DEBUG: ✓ FAISS index loaded successfully")
        
        # Test the index
        test_docs = db.similarity_search("test", k=1)
        print(f"DEBUG: FAISS index test - found {len(test_docs)} documents")
    except Exception as e:
        print(f"DEBUG: ✗ Failed to load FAISS index: {e}")
        st.error(f"Failed to load FAISS index: {e}")
        raise

    # Load LLM
    try:
        print("DEBUG: Loading text2text-generation pipeline (flan-t5-base)...")
        llm_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=150,
            device=device_id
           ## "text2text-generation",
           ## model="google/flan-t5-base",
            ## max_length=150,
            ## device=device_id
        )
        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        print("DEBUG: ✓ LLM pipeline loaded successfully")
        
        # Test the LLM
        test_response = llm.invoke("Hello")
        print(f"DEBUG: LLM test response: {test_response[:50]}...")
    except Exception as e:
        print(f"DEBUG: ✗ Failed to load LLM pipeline: {e}")
        st.error(f"Failed to load LLM pipeline: {e}")
        raise

    # Load sentiment analysis
    try:
        print("DEBUG: Loading sentiment-analysis pipeline...")
        sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device_id
        )
        sentiment_labels = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
        print("DEBUG: ✓ Sentiment pipeline loaded successfully")
        
        # Test sentiment
        test_sentiment = sentiment_pipe("This is great!")
        print(f"DEBUG: Sentiment test: {test_sentiment}")
    except Exception as e:
        print(f"DEBUG: ✗ Failed to load sentiment pipeline: {e}")
        st.error(f"Failed to load sentiment pipeline: {e}")
        raise
    
    print("DEBUG: === load_rag_pipeline END ===")
    return db, llm, embeddings, sentiment_pipe, sentiment_labels

def predict_emotion(processor, model, device, image: Image.Image):
    """Predicts the emotion from a PIL Image."""
    print("DEBUG: === predict_emotion START ===")
    
    try:
        # Ensure RGB format
        if image.mode != "RGB":
            print(f"DEBUG: Converting image from {image.mode} to RGB")
            image = image.convert("RGB")
        
        print(f"DEBUG: Processing image of size {image.size}")
        inputs = processor(images=image, return_tensors="pt").to(device)
        print(f"DEBUG: Image processed, input shape: {inputs['pixel_values'].shape}")
        
        with torch.no_grad():
            logits = model(**inputs).logits
            print(f"DEBUG: Model inference complete, logits shape: {logits.shape}")
        
        predicted_class_id = logits.argmax(-1).item()
        emotion = model.config.id2label[predicted_class_id]
        confidence = torch.softmax(logits, dim=-1).max().item()
        
        print(f"DEBUG: ✓ Emotion predicted: {emotion} (confidence: {confidence:.2f})")
        print("DEBUG: === predict_emotion END ===")
        return emotion
    
    except Exception as e:
        print(f"DEBUG: ✗ Error in predict_emotion: {e}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        raise

def query_rag_simple(db, llm, query):
    """
    Simplified RAG query without using the deprecated RetrievalQA chain.
    This function directly retrieves documents and generates answers.
    """
    print(f"DEBUG: === query_rag_simple START ===")
    print(f"DEBUG: Query: {query}")
    
    try:
        # Step 1: Retrieve relevant documents
        print(f"DEBUG: Searching for relevant documents...")
        docs = db.similarity_search(query, k=4)
        print(f"DEBUG: ✓ Retrieved {len(docs)} documents")
        
        if not docs:
            print("DEBUG: ⚠ No documents found")
            return {
                "result": "No relevant documents found for your query.",
                "source_documents": []
            }
        
        # Step 2: Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        print(f"DEBUG: Built context from {len(docs)} documents, total length: {len(context)} chars")
        
        # Step 3: Create prompt
        prompt = PROMPT_TEMPLATE.format(context=context, question=query)
        print(f"DEBUG: Created prompt, length: {len(prompt)} chars")
        
        # Step 4: Generate answer
        print("DEBUG: Generating answer with LLM...")
        answer = llm.invoke(prompt)
        print(f"DEBUG: ✓ Answer generated, length: {len(answer)} chars")
        
        print("DEBUG: === query_rag_simple END ===")
        return {
            "result": answer,
            "source_documents": docs
        }
    
    except Exception as e:
        print(f"DEBUG: ✗ Error in query_rag_simple: {e}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return {
            "result": f"Error processing query: {str(e)}",
            "source_documents": []
        }

def analyze_sentiment(sentiment_pipe, labels_map, text):
    """Analyzes and formats the sentiment of a text."""
    print(f"DEBUG: === analyze_sentiment START ===")
    
    try:
        print(f"DEBUG: Analyzing sentiment for text (first 50 chars): {text[:50]}...")
        sentiment = sentiment_pipe(text)[0]
        
        # Debug: Print raw sentiment output
        print(f"DEBUG: Raw sentiment output: {sentiment}")
        
        # Handle both LABEL_X and direct label names
        raw_label = sentiment['label']
        
        # First try the labels_map
        if raw_label in labels_map:
            label = labels_map[raw_label]
        # Then check if it's already a readable label
        elif raw_label.upper() in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            label = raw_label.upper()
        # Fallback mapping for newer model versions
        elif raw_label.lower() == 'positive':
            label = 'POSITIVE'
        elif raw_label.lower() == 'negative':
            label = 'NEGATIVE'
        elif raw_label.lower() == 'neutral':
            label = 'NEUTRAL'
        else:
            label = f'UNKNOWN ({raw_label})'
            
        score = sentiment['score']
        result = f"{label} (Score: {score:.2f})"
        
        print(f"DEBUG: ✓ Sentiment result: {result}")
        print("DEBUG: === analyze_sentiment END ===")
        return result
    
    except Exception as e:
        print(f"DEBUG: ✗ Error in sentiment analysis: {e}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return "Error analyzing sentiment"

# Compatibility wrapper for old code
def query_rag(qa_chain, query):
    """
    Compatibility wrapper to work with existing code.
    qa_chain should be a tuple: (db, llm, embeddings)
    """
    print("DEBUG: === query_rag (compatibility wrapper) ===")
    
    if isinstance(qa_chain, tuple) and len(qa_chain) >= 2:
        db, llm = qa_chain[:2]
        return query_rag_simple(db, llm, query)
    else:
        print(f"DEBUG: ⚠ Unexpected qa_chain type: {type(qa_chain)}")
        return {
            "result": "Error: Invalid chain configuration",
            "source_documents": []
        }

print("DEBUG: model_utils.py fully loaded and ready!")
