import os
import re
import sys
import json
import urllib.request
import urllib.error
import numpy as np
from sauti_rag import SautiEngine

SLANG_DB = {
    "niaje": {"doc_id": "doc_sheng_1", "text": "Sheng: 'Niaje' / 'Niaje bana' means 'How are you' or 'What's up'"},
    "form": {"doc_id": "doc_sheng_2", "text": "Sheng: 'Form ni gani' / 'Form' translates to 'What is the plan?'"},
    "poa": {"doc_id": "doc_sheng_3", "text": "Sheng/Swahili: 'Poa' / 'Fiti' translates to 'Cool / Fine / Good / Okay'"},
    "noma": {"doc_id": "doc_sheng_4", "text": "Sheng: 'Noma' translates to 'Dope / Tough / Problem / Crazy / Hard'"},
    "mbogi": {"doc_id": "doc_sheng_5", "text": "Sheng: 'Mbogi' translates to 'Crew / Squad / Group of friends'"},
    "anguka nayo": {"doc_id": "doc_sheng_6", "text": "Sheng: 'Anguka nayo' translates to 'Go with it / Jam to it / Dance enthusiastically'"},
    "ukoje": {"doc_id": "doc_sheng_7", "text": "Sheng/Swahili: 'Ukoje' translates to 'How are you?' or 'How are you doing?'"},
    "vipi": {"doc_id": "doc_sheng_8", "text": "Sheng: 'Vipi' / 'Vipi kaka' means 'How are things?' or 'How is it?'"},
    "habari": {"doc_id": "doc_swahili_1", "text": "Swahili: 'Habari za asubuhi' translates to 'Good morning'"},
    "unapendeza": {"doc_id": "doc_swahili_2", "text": "Swahili: 'Unapendeza' translates to 'You look great / You are attractive'"},
    "cheki": {"doc_id": "doc_sheng_9", "text": "Sheng: 'Cheki' translates to 'Look / Check this out / See'"},
    "koma": {"doc_id": "doc_sheng_10", "text": "Sheng/Swahili: 'Koma' translates to 'Stop / Cease / Desist'"}
}

KEYS = list(SLANG_DB.keys())
DIM = len(KEYS)

engine = SautiEngine()
for idx, key in enumerate(KEYS):
    vec = [0.0] * DIM
    vec[idx] = 1.0
    engine.add_document(SLANG_DB[key]["doc_id"], SLANG_DB[key]["text"], vec)

ACTIVE_MODEL_CACHE = None

def text_to_vector(text):
    t = text.lower().strip()
    vec = [0.0] * DIM
    for idx, key in enumerate(KEYS):
        if key in t:
            vec[idx] = 1.0
            return vec
    return vec

def calculate_cosine_similarity(v1, v2):
    a, b = np.array(v1, dtype=float), np.array(v2, dtype=float)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def discover_groq_model(api_key):
    global ACTIVE_MODEL_CACHE
    if ACTIVE_MODEL_CACHE:
        return ACTIVE_MODEL_CACHE

    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Mozilla/5.0 (SautiNLP/1.0 Client)",
        "Accept": "application/json"
    }

    PREFERRED_CHATS = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-specdec",
        "mixtral-8x7b-32768"
    ]

    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            available_ids = [m["id"] for m in data.get("data", [])]

            for pref in PREFERRED_CHATS:
                if pref in available_ids:
                    ACTIVE_MODEL_CACHE = pref
                    return ACTIVE_MODEL_CACHE

            valid = [
                m for m in available_ids 
                if not any(x in m.lower() for x in [
                    "whisper", "vision", "guard", "orpheus", "tts", "arabic", 
                    "canopy", "oss", "qwen", "deepseek", "r1", "reason"
                ])
            ]
            if valid:
                ACTIVE_MODEL_CACHE = valid[0]
                return ACTIVE_MODEL_CACHE
    except Exception:
        pass

    ACTIVE_MODEL_CACHE = "llama-3.1-8b-instant"
    return ACTIVE_MODEL_CACHE

def clean_reasoning_output(text):
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
    return cleaned.strip()

def query_groq_llm(user_query, retrieved_context=""):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "[Groq Offline] Set GROQ_API_KEY to activate AI definitions."

    model_name = discover_groq_model(api_key)
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    system_prompt = (
        "You are SautiNLP, an expert dictionary for Kenyan Sheng and Swahili.\n"
        "Instructions:\n"
        "1. Define the user's phrase clearly in plain text.\n"
        "2. Provide 1 concise sentence translation and 1 simple example sentence.\n"
        "3. Do NOT show reasoning or scratchpad output."
    )
    
    user_content = f"Phrase: '{user_query}'"
    if retrieved_context.strip():
        user_content += f"\nMatched Context: {retrieved_context}"
    else:
        user_content += "\nNote: Not found in local dictionary. Provide standard Kenyan Sheng/Swahili translation."

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.2,
        "max_tokens": 200
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (SautiNLP/1.0 Client)",
        "Accept": "application/json"
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            cleaned = clean_reasoning_output(content)
            return f"[{model_name}] {cleaned}" if cleaned else "[Unable to translate phrase]"
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode('utf-8')
        return f"[Groq HTTP {e.code}]: {err_msg}"
    except Exception as e:
        return f"[Groq Error]: {str(e)}"

def run_cli():
    print("==================================================")
    print(" SautiNLP Interactive CLI (Dynamic RAG + Groq)")
    print(" Type any Sheng/Swahili phrase or 'exit'")
    print("==================================================")
    while True:
        try:
            query = input("\nsauti-nlp> ").strip()
            if not query or query.lower() == "exit":
                break
            
            vec = text_to_vector(query)
            raw_matches = engine.retrieve_context(vec, top_k=2)
            context_str = ""
            
            print(f"\n[Query]: \"{query}\"")
            for item in raw_matches:
                doc_id = item[0] if isinstance(item, (list, tuple)) else item.get("id", "doc")
                text = item[1] if isinstance(item, (list, tuple)) else item.get("text", "")
                
                matching_entry = next((v for k, v in SLANG_DB.items() if v["doc_id"] == doc_id), None)
                if matching_entry:
                    key_idx = KEYS.index(next(k for k, v in SLANG_DB.items() if v["doc_id"] == doc_id))
                    target_vec = [0.0] * DIM
                    target_vec[key_idx] = 1.0
                    score = calculate_cosine_similarity(vec, target_vec)
                    
                    if score >= 0.99:
                        print(f"  • [RAG Context]: {text} (Similarity: {score:.4f})")
                        context_str += f"{text}; "

            print("  • [Groq AI Response]:")
            ai_out = query_groq_llm(query, context_str)
            print(f"    {ai_out}")
        except Exception as e:
            print(f"[!] Error: {e}")

if __name__ == "__main__":
    run_cli()
