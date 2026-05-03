# InsureQ — LLM Document Processing System

> End-to-end insurance document intelligence: ingestion → semantic search → LLM Q&A → REST API → Streamlit UI

**Status: Phase 1–4 Complete (100%)**

---

## 🌍 Real-World Importance

Insurance policies are notoriously dense — hundreds of pages of legal clauses, exclusions, waiting periods, and sub-limits that most people never fully read or understand. This creates a massive information gap between insurers and policyholders, leading to:

- **Claim denials** because customers didn't know about exclusions buried on page 47
- **Delayed decisions** — agents spend hours manually searching policy documents to answer a single question
- **Mis-selling** — customers buy policies without understanding what's actually covered
- **Accessibility barriers** — non-native speakers and elderly users struggle with complex policy language

InsureQ directly addresses this by turning any insurance document into a conversational AI assistant. A user can upload their policy PDF and instantly ask *"Is my knee surgery covered?"* or *"What is the waiting period for diabetes?"* and get a precise, source-cited answer in seconds — not hours.

**Who benefits:**
- Policyholders who want to understand what they're paying for
- Insurance agents who need fast, accurate policy lookups
- Claims teams who need to verify coverage eligibility quickly
- InsurTech companies building customer-facing self-service tools
- Compliance teams auditing policy language across large document sets

---

## 🚀 What It Does

InsureQ processes insurance policy documents (PDF/DOCX) and lets you ask natural language questions about them. It uses a Retrieval-Augmented Generation (RAG) pipeline backed by ChromaDB and Qwen2.5-72B via the HuggingFace Inference API — no local GPU required for the LLM.

---

## 📁 Project Structure

```
llm_document_processing/
├── app.py                          # Streamlit UI (InsureQ)
├── config.yaml                     # Central configuration
├── requirements.txt                # All dependencies (Phase 1–4)
├── check_models.py                 # Utility to test available HF models
├── Dockerfile
├── docker-compose.yml
├── .env                            # HF_TOKEN (gitignored)
│
├── data/
│   ├── raw_documents/              # 10 PDFs + 5 DOCX base corpus (doc1–doc15)
│   ├── user_documents/             # Per-session uploaded files (cleared on API restart)
│   ├── processed/                  # chunked_documents.json, embeddings, etc.
│   └── vector_db/chroma_db/        # ChromaDB persistent storage
│
├── logs/                           # Runtime logs
│
├── src/
│   ├── __init__.py
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
project:
  name: "LLM Document Processing System"
  version: "1.0.0"

paths:
  raw_documents: "./data/raw_documents/"
  user_documents: "./data/user_documents/"
  processed_data: "./data/processed/"
  vector_db: "./data/vector_db/chroma_db/"

document_processing:
  supported_formats: [".pdf", ".docx", ".eml", ".txt"]
  chunk_method: "clause"       # sentence | paragraph | hybrid | clause
  chunk_size: 200
  chunk_overlap: 1
  max_file_size_mb: 50
  cleaning_patterns:
    remove_headers: true
    remove_footers: true
    normalize_whitespace: true
    preserve_medical_terms: true

embeddings:
  model_name: "BAAI/bge-large-en-v1.5"
  batch_size: 32
  normalize: true
  instruction_prefix: "Represent this document for retrieval: "
  query_instruction_prefix: "Represent this question for searching relevant passages: "

vector_db:
  provider: "chromadb"
  collection_name: "document_chunks"
  user_collection_name: "user_chunks"
  similarity_metric: "cosine"

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

performance:
  cache_embeddings: true
  cache_ttl: 3600
  enable_gpu: true
  max_batch_size: 32
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

> On every API startup, the `user_chunks` ChromaDB collection and `data/user_documents/` folder are automatically cleared so uploaded files don't persist across sessions.

---

## 🔍 Check Available HF Models

`check_models.py` tests a list of HuggingFace Inference API models against your token to see which ones are accessible:

```bash
python check_models.py
```

Tested candidates include: `Qwen2.5-72B-Instruct`, `Qwen2.5-7B-Instruct`, `Mistral-7B-Instruct`, `Llama-3-8B-Instruct`, `Phi-3-mini`, `Gemma-2-2b-it`, and more. Use this to find a working model if the default is rate-limited.

---

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API status, model info, version |
| POST | `/query` | LLM-powered Q&A with source attribution |
| POST | `/search` | Semantic search (no LLM) |
| POST | `/upload` | Upload and process a new document (saved to `user_documents/`, indexed into `user_chunks`) |
| POST | `/reindex` | Rebuild ChromaDB vector index from all processed documents |
| GET | `/stats` | Collection stats (chunk count, embedding model, similarity metric) |

Interactive docs: `http://localhost:8000/docs` (Swagger UI)

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
- Conversational chat interface with styled bubbles
- Intent badge on every response (`coverage_check`, `exclusion_check`, `waiting_period`, etc.)
- Collapsible source attribution — document name, page, section, similarity score
- Response latency displayed per message
- Clear conversation button

**🔍 Semantic Search**
- Direct vector similarity search without LLM
- Animated score bar per result
- Shows chunk text, file name, page number, chunk ID, similarity score

**📁 Document Management**
- Upload PDF/DOCX — auto-processed and indexed into `user_chunks` collection
- Rebuild ChromaDB index with one click (`/reindex`)
- Live index stats: total chunks, collection name, embedding model, similarity metric

**Sidebar**
- API online/offline status pill
- LLM model and embedding model info
- Total indexed chunks counter
- Top K slider (1–15, controls both Chat and Search)

---

## 🐳 Docker

```bash
# Build and run
docker-compose up --build

# API available at http://localhost:8000
```

`docker-compose.yml` mounts `./data` and `./logs` as volumes so documents and the vector index persist across container restarts. The `.env` file is passed automatically for the HuggingFace token.

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Activate venv: `.\venv\Scripts\activate` then `pip install -r requirements.txt` |
| `Collection not found` | Run Phase 2 first: `python src/phase2_semantic_search.py` |
| `API Offline` in UI | Start API from project root: `python src/phase4_api.py` |
| `hf_token` empty error | Add `HF_TOKEN=your_token` to `.env` file |
| Model rate-limited / 503 | Run `python check_models.py` to find an available model, update `config.yaml` |
| Low similarity scores | Delete ChromaDB and rebuild: `python src/phase2_semantic_search.py --force` |
| GPU not detected | Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| Uploaded file not found after restart | Expected — `user_documents/` and `user_chunks` are cleared on every API startup by design |

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
Insurance Documents (PDF / DOCX)
        │
        ▼
 Phase 1 — Text Extraction & Chunking
 PyMuPDF / python-docx → per-page text + metadata
 Chunking: clause | sentence | paragraph | hybrid
        │
        ▼
 Phase 2 — Embedding & Vector Index
 BAAI/bge-large-en-v1.5 → ChromaDB (cosine similarity)
 Collections: document_chunks (base) + user_chunks (uploads)
        │
        ▼
 Phase 2 — Semantic Search + Optional Reranker
 cross-encoder/ms-marco-MiniLM-L-6-v2
        │
        ▼
 Phase 3 — Intent Classification
 coverage_check | exclusion_check | waiting_period | ...
        │
        ▼
 Phase 3 — LLM Reasoning
 Qwen/Qwen2.5-72B-Instruct via HuggingFace Inference API
        │
        ▼
 Structured Answer + Source Attribution
        │
        ├──▶ Phase 4 — FastAPI REST API
        │    /query /search /upload /reindex /stats /health
        │    Swagger UI at :8000/docs
        │
        └──▶ Phase 4 — Streamlit UI (InsureQ)
             💬 Chat Q&A | 🔍 Semantic Search | 📁 Document Management
             http://localhost:8501
```
