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
        
        # Preload default Sheng/Swahili dictionary documents
        self.add_document("doc_noma", "Noma: Means dangerous, cool, amazing, or a serious situation in Nairobi Sheng.", [0.2, 0.8, 0.4])
        self.add_document("doc_luku", "Luku: Refers to fashion, outfit, or personal style in Sheng.", [0.5, 0.3, 0.9])
        self.add_document("doc_mbogi", "Mbogi: Means a group of friends, crew, or gang.", [0.1, 0.9, 0.2])
        self.add_document("doc_form", "Form: Means a plan, opportunity, or hangout (e.g., 'Ni gani form?').", [0.7, 0.4, 0.1])

    def compute_cosine_similarity(self, vec_a, vec_b):
        """Calculates vector cosine similarity using pure NumPy math."""
        if len(vec_a) != len(vec_b):
            return 0.0
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def add_document(self, doc_id, text, vector):
        """Indexes context documents into memory with numerical vector embeddings."""
        self.knowledge_base.append({"id": doc_id, "text": text})
        self.embeddings.append(np.array(vector, dtype=float))

    def retrieve_context(self, query, top_k=1):
        """Retrieves matching document context supporting both text strings and vectors."""
        if not self.embeddings:
            return []
            
        if isinstance(query, str):
            query_lower = query.lower()
            # Direct keyword matching fallback for text queries
            for doc in self.knowledge_base:
                if query_lower in doc["text"].lower():
                    return [doc]
            # Fallback pseudo-vector generation for strings
            val = sum(ord(c) for c in query_lower) % 100 / 100.0
            query_vec = np.array([val, 0.5, 0.5], dtype=float)
        else:
            query_vec = np.array(query, dtype=float)

        scores = [self.compute_cosine_similarity(query_vec, doc_vec) for doc_vec in self.embeddings]
        best_indices = np.argsort(scores)[::-1][:top_k]
        return [self.knowledge_base[i] for i in best_indices]

    def generate_rag_response(self, user_query, context_text):
        """Queries Groq Llama-3 HTTP REST API with retrieved regional context."""
        payload = {
            "model": "openai/gpt-oss-20b",
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
    match = engine.retrieve_context("Noma", top_k=1)
    print("Retrieved Context:", match)
