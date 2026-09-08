import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sauti_rag import SautiEngine

def test_cosine_similarity():
    engine = SautiEngine()
    vec_a = [1.0, 0.0]
    vec_b = [1.0, 0.0]
    similarity = engine.compute_cosine_similarity(vec_a, vec_b)
    assert abs(similarity - 1.0) < 1e-5

def test_context_retrieval():
    engine = SautiEngine()
    engine.add_document("doc1", "Sheng dialect entry", [0.9, 0.1])
    engine.add_document("doc2", "Swahili formal entry", [0.1, 0.9])
    results = engine.retrieve_context([0.85, 0.15], top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
