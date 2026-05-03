# InsureQ — LLM Document Processing System

> End-to-end insurance document intelligence: ingestion → semantic search → LLM Q&A → REST API → Streamlit UI

**Status: Phase 1–4 Complete (100%)**

---

## 🚀 What It Does

InsureQ processes insurance policy documents (PDF/DOCX) and lets you ask natural language questions about them. It uses a retrieval-augmented generation (RAG) pipeline backed by ChromaDB and Qwen2.5-72B.

---

## 📁 Project Structure

```
llm_document_processing/
├── app.py                          # Streamlit UI (InsureQ)
├── config.yaml                     # Central configuration
├── requirements.txt                # All dependencies (Phase 1–4)
├── Dockerfile
├── docker-compose.yml
├── .env                            # HF_TOKEN (gitignored)
│
├── data/
│   ├── raw_documents/              # Place PDF/DOCX files here
│   ├── processed/                  # chunked_documents.json, embeddings, etc.
│   └── vector_db/chroma_db/        # ChromaDB persistent storage
│
├── logs/                           # Runtime logs
│
├── src/
│   ├── phase1_document_processing.py
│   ├── phase2_semantic_search.py
│   ├── phase3_llm_engine.py
│   └── phase4_api.py               # FastAPI REST API
│
└── notebooks/
    ├── Phase1.ipynb
    └── Phase2.ipynb
```

---

## 🛠 Installation

```bash
# 1. Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your HuggingFace token in .env
HF_TOKEN=your_token_here
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
document_processing:
  chunk_method: "clause"       # sentence | paragraph | hybrid | clause
  chunk_size: 200
  chunk_overlap: 1

embeddings:
  model_name: "BAAI/bge-large-en-v1.5"
  instruction_prefix: "Represent this document for retrieval: "
  query_instruction_prefix: "Represent this question for searching relevant passages: "

semantic_search:
  top_k: 7
  similarity_threshold: 0.0
  rerank: false
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

llm:
  provider: "huggingface"
  model: "Qwen/Qwen2.5-72B-Instruct"
  hf_token: ""                 # leave empty — loaded from .env automatically
  temperature: 0.0
  max_tokens: 512
  context_token_limit: 3000

api:
  host: "0.0.0.0"
  port: 8000
  reload: false
  workers: 1
```

> `hf_token` in config can be left empty. The system automatically loads `HF_TOKEN` from `.env` via `python-dotenv`.

---

## 🏃 Quick Start

### Step 1 — Process documents
```bash
python src/phase1_document_processing.py
```

### Step 2 — Build semantic search index
```bash
python src/phase2_semantic_search.py
# Force rebuild:
python src/phase2_semantic_search.py --force
```

### Step 3 — Test LLM engine (CLI)
```bash
python src/phase3_llm_engine.py
```

### Step 4 — Start REST API
```bash
# Run from project root (not from src/)
python src/phase4_api.py
```

### Step 5 — Launch Streamlit UI
```bash
# In a second terminal (API must be running)
streamlit run app.py
```

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API status, model info, version |
| POST | `/query` | LLM-powered Q&A with source attribution |
| POST | `/search` | Semantic search (no LLM) |
| POST | `/upload` | Upload and process a new document |
| POST | `/reindex` | Rebuild ChromaDB vector index |
| GET | `/stats` | Collection stats |

Interactive docs available at `http://localhost:8000/docs` (Swagger UI).

### Example: Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Is knee surgery covered?", "top_k": 7}'
```

```json
{
  "query": "Is knee surgery covered?",
  "intent": "coverage_check",
  "answer": "...",
  "sources": [...],
  "latency_ms": 1234.5
}
```

---

## 🖥️ Streamlit UI (InsureQ)

Launch with `streamlit run app.py` while the API is running. Opens at `http://localhost:8501`.

### Pages

**💬 Chat Q&A**
- Conversational interface with chat bubbles
- Intent badge on every response (coverage_check, exclusion_check, etc.)
- Collapsible source attribution with document name, page, section, and similarity score
- Response latency displayed per message
- Clear conversation button

**🔍 Semantic Search**
- Direct vector similarity search without LLM
- Animated score bar per result
- Shows chunk text, file name, page number, chunk ID

**📁 Document Management**
- Upload PDF/DOCX files — auto-processed and indexed
- Rebuild ChromaDB index with one click
- Live index stats (chunk count, collection name, embedding model, similarity metric)

**Sidebar**
- API online/offline status pill
- LLM model and embedding model info
- Total indexed chunks counter
- Top K slider (controls results for both Chat and Search)

---

## 🐳 Docker

```bash
# Build and run
docker-compose up --build

# API available at http://localhost:8000
```

`docker-compose.yml` mounts `./data` and `./logs` as volumes so documents and logs persist across restarts. The `.env` file is passed automatically for the HuggingFace token.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Activate venv: `.\venv\Scripts\activate` then `pip install -r requirements.txt` |
| `Collection not found` | Run Phase 2 first: `python src/phase2_semantic_search.py` |
| `API Offline` in UI | Start API from project root: `python src/phase4_api.py` |
| `hf_token` empty error | Add `HF_TOKEN=your_token` to `.env` file |
| Low similarity scores | Delete ChromaDB and rebuild: `python src/phase2_semantic_search.py --force` |
| GPU not detected | Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |

---

## 📊 Embedding Model Options

| Model | Params | Embedding Dim | VRAM |
|-------|--------|---------------|------|
| bge-small-en-v1.5 | 33M | 384 | < 1 GB |
| bge-base-en-v1.5 | 109M | 768 | ~1 GB |
| bge-large-en-v1.5 | 335M | 1024 | ~2 GB |

To switch models: update `embeddings.model_name` in `config.yaml`, delete `data/vector_db/chroma_db/`, then run Phase 2 with `--force`.

---

## 🖼️ Architecture

```
Insurance Documents
        │
        ▼
 Text Extraction (Phase 1)
 PDF/DOCX → per-page text + metadata
        │
        ▼
 Document Chunking (Phase 1)
 clause / sentence / paragraph / hybrid
        │
        ▼
 Embedding Generation (Phase 2)
 BAAI/bge-large-en-v1.5
        │
        ▼
 ChromaDB Vector Store (Phase 2)
        │
        ▼
 Semantic Search + Reranker (Phase 2)
        │
        ▼
 Intent Classification (Phase 3)
 coverage_check | exclusion | waiting_period | ...
        │
        ▼
 LLM Reasoning — Qwen2.5-72B (Phase 3)
 HuggingFace Inference API
        │
        ▼
 Structured Answer + Sources (Phase 3)
        │
        ├──▶ FastAPI REST API (Phase 4)
        │    /query /search /upload /reindex /stats /health
        │
        └──▶ Streamlit UI — InsureQ (Phase 4)
             Chat Q&A | Semantic Search | Document Management
```
