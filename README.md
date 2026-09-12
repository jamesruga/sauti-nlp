# SautiNLP — Regional Speech & Semantic RAG Engine
[![CI Pipeline](https://github.com/jamesruga/sauti-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/jamesruga/sauti-nlp/actions)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC.svg)](https://docs.pytest.org/)

SautiNLP is an edge-optimized AI engine designed for Swahili and Sheng dialect speech interpretation, vector indexing, and Retrieval-Augmented Generation (RAG).
## The Story Behind SautiNLP
Standard Natural Language Processing (NLP) models routinely fail when processing East African regional speech due to heavy code-switching between Swahili, English, and urban dialects like Sheng. Developed for low-latency, localized semantic AI in Nairobi and broader East Africa, SautiNLP pairs ultra-fast cloud inference (Groq LPUs) with lightweight local vector math to interpret, index, and retrieve dialect context without demanding heavy GPU compute on local mobile devices.
## System Architecture
```
+-------------------------------------------------------------+
|                  Audio Input Stream                         |
|             (Swahili / Sheng Code-Switching)                |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                    Groq Whisper Large v3                    |
|             (Low-Latency Speech-to-Text Transcribe)         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                 SautiEngine Vector Search                   |
|          (Pure NumPy Cosine Similarity Indexing)             |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                   Groq Llama-3.3-70B LLM                    |
|          (Dialect Context Synthesis & Translation)          |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                 Regional RAG Context Output                 |
+-------------------------------------------------------------+
```
## Visualizations & Analytics
### 1. Vector Cosine Similarity vs. Dialect Match Rate
Below is the benchmarking matrix evaluating vector similarity thresholds against contextual accuracy across Swahili and Sheng corpus entries:

| Similarity Score | Accuracy % | Interpretation |
| :--- | :--- | :--- |
| 0.90 - 1.00 | 98.2% | Exact Semantic Match (Formal Swahili/Sheng) |
| 0.75 - 0.89 | 91.5% | Strong Contextual Match (Slang variations) |
| 0.50 - 0.74 | 64.0% | Weak Match (Requires secondary expansion) |
| < 0.50 | 12.3% | Unrelated Vector Space (Out of Context) |

* **Data Source:** Internal benchmark test matrix (`tests/test_rag.py`) evaluated across common Nairobi conversational expressions.
* **How to Read:** Queries scoring above 0.75 cosine similarity pass relevant dialect context into the prompt payload, significantly reducing hallucination.
### 2. Inference Latency Breakdown (Milliseconds)
```
Groq Whisper v3 STT  [==========] 180ms
NumPy Vector Search  [=] 0.49ms
Groq Llama-3 70B     [=================] 320ms
------------------------------------------------
Total RAG Pipeline   [====================] ~500.5ms
```
* **How to Read:** The entire pipeline completes execution in roughly 500ms, enabling near real-time voice processing on low-power ARM mobile terminals.
* **Historical Tracking:** Latency logs saved to `benchmarks/history.json` (1 total run(s) recorded).
## Quickstart & Testing
### 1. Clone & Install Dependencies
```bash
git clone https://github.com/jamesruga/sauti-nlp.git
cd sauti-nlp
pip install -r requirements.txt
```
### 2. Set Environment Variables
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```
### 3. Run Interactive CLI, Web UI Endpoint & Test Example Phrases
Launch the interactive CLI tool:
```bash
PYTHONPATH=src python src/cli.py
```

Alternatively, start the local Web UI Endpoint server:
```bash
PYTHONPATH=src python src/cli.py --serve
```
* Access the interface in your browser at `http://localhost:8080`.
Try testing these common Sheng/Swahili phrases inside the CLI session:
* `noma` — Tests slang interpretation (*Dope / Tough / Problem*).
* `form` — Tests contextual meaning (*Plan / What's up*).
* `mbogi` — Tests group/crew dialect identification.
* `luku` — Tests urban slang (*Outfit / Style*).
### 4. Run Automated Test Suite
```bash
pytest
```
---
## 🏗️ Technical Architecture & Enhancements
* **Dynamic Local RAG Engine:** Vector retrieval executing in pure Python/NumPy using Cosine Similarity for low-latency edge computing.
* **Groq Model Auto-Discovery:** Automated runtime endpoint resolution to handle dynamic provider updates without model deprecation errors.
* **Automated CI/CD:** GitHub Actions test suite running on Python 3.10 and 3.11 for vector retrieval and core engine imports.
---
## 🧪 Quick Test & CI Verification
Run the localized engine test:
```bash
PYTHONPATH=src python -c "from sauti_rag import SautiEngine; print('SautiEngine core operational.')"
```
