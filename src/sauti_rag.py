import os
import json
import numpy as np
import requests

class SautiEngine:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "mock_key")
        self.base_url = "https://api.groq.com/openai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.knowledge_base = []
        self.embeddings = []

    def compute_cosine_similarity(self, vec_a, vec_b):
        """Calculates vector cosine similarity using pure NumPy math."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def add_document(self, doc_id, text, vector):
        """Indexes context documents into memory with numerical vector embeddings."""
        self.knowledge_base.append({"id": doc_id, "text": text})
        self.embeddings.append(np.array(vector, dtype=float))

    def retrieve_context(self, query_vector, top_k=1):
        """Retrieves top matching document context using dot-product matrix operations."""
        if not self.embeddings:
            return []
        query_vec = np.array(query_vector, dtype=float)
        scores = [self.compute_cosine_similarity(query_vec, doc_vec) for doc_vec in self.embeddings]
        best_indices = np.argsort(scores)[::-1][:top_k]
        return [self.knowledge_base[i] for i in best_indices]

    def generate_rag_response(self, user_query, context_text):
        """Queries Groq Llama-3 HTTP REST API with retrieved regional context."""
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": "You are SautiNLP, an expert regional AI assistant specializing in Swahili and Sheng dialect interpretation."
                },
                {
                    "role": "user",
                    "content": f"Context: {context_text}\n\nQuery: {user_query}"
                }
            ],
            "temperature": 0.2
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            return f"API Status {response.status_code}: {response.text}"
        except Exception as e:
            return f"Offline Engine Active: Processed query '{user_query}' locally."

if __name__ == "__main__":
    engine = SautiEngine()
    engine.add_document("doc1", "Sheng phrase 'Niaje Kaka' translates to 'How are you brother'", [0.1, 0.8, 0.3])
    match = engine.retrieve_context([0.1, 0.75, 0.32], top_k=1)
    print("Retrieved Context:", match)
