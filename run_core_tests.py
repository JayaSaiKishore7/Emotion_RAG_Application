import traceback
import core.models as m
from PIL import Image

print("=== START TESTS ===")

# Test 1: emotion model (may download/load; will reuse cache)
try:
    print("\n--- Test: load_emotion_model() ---")
    proc, model, device = m.load_emotion_model()
    print("Loaded emotion model. device:", device)
    img = Image.new("RGB", (224,224), color=(255,255,255))
    emotion = m.predict_emotion(proc, model, device, img)
    print("predict_emotion result:", emotion)
except Exception as e:
    print("ERROR in emotion model test:", e)
    traceback.print_exc()

 #Test 2: RAG pipelines
try:
    print("\n--- Test: load_rag_pipeline() ---")
    db, llm, embeddings, sentiment_pipe, sentiment_labels = m.load_rag_pipeline()
    print("Loaded RAG pipeline. DB type:", type(db))
    print("Running a quick query_rag_simple/test search...")
    try:
        r = m.query_rag_simple(db, "test query")
        print("query_rag_simple result (first 200 chars):")
        print(r.get("result","<no result>")[:200])
    except Exception as e:
        print("query_rag_simple failed:", e)
        # fallback to query_rag if available
        try:
            r2 = m.query_rag((db, llm, embeddings), "test query")
            print("query_rag result (first 200 chars):")
            print(r2.get("result","<no result>")[:200])
        except Exception as ex:
            print("query_rag also failed:", ex)
            traceback.print_exc()
except Exception as e:
    print("ERROR in RAG pipeline test:", e)
    traceback.print_exc()

# Test 3: Sentiment
try:
    print("\n--- Test: analyze_sentiment() ---")
    s = m.analyze_sentiment(sentiment_pipe, sentiment_labels, "This is a wonderful product!")
    print("analyze_sentiment result:", s)
except Exception as e:
    print("ERROR in sentiment test:", e)
    traceback.print_exc()

print("\n=== TESTS COMPLETE ===")
