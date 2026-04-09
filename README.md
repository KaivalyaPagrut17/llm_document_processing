# LLM Document Processing System - Phase 1–2 Implementation (50% Progress)

## 🚀 Project Overview

This project implements the first half of an LLM-based document processing system.

**Currently implemented modules:**

• Document ingestion and preprocessing
• Text extraction from PDF/DOCX files
• Document chunking
• Embedding generation
• Vector database storage using ChromaDB
• Semantic search for retrieving relevant clauses

**Modules planned for next phase:**

• Query intent analysis
• LLM reasoning engine
• Structured decision generation
• REST API interface

## 📁 Project Structure

```
llm_document_processing/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.yaml                    # Main configuration
├── .env                          # Environment variables (create this)
├── .gitignore                    # Git ignore file
│
├── data/                         # All data files
│   ├── raw_documents/           # Place your PDF/DOCX/EML files here
│   ├── processed/               # Generated processed files
│   └── vector_db/               # ChromaDB storage
│
├── src/                         # Main source code
│   ├── phase1_document_processing.py
│   └── phase2_semantic_search.py
│
└── notebooks/                   # Jupyter notebooks
    ├── Phase1.ipynb
    └── Phase2.ipynb
```

## 🛠 Installation & Setup

### Chunking options

The document processor now supports three chunking strategies:

* **sentence** (default) – group one or more sentences into each chunk, with a
  configurable word budget and sentence overlap.
* **paragraph** – split text on blank lines, producing one chunk per
  paragraph.
* **hybrid** – break into paragraphs and then sentence‑chunk long paragraphs.

Configure the method and parameters under `document_processing` in
`config.yaml` (see the file for defaults).


### Step 1: Environment Setup

```bash
# Clone or create the project directory
mkdir llm_document_processing
cd llm_document_processing

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configuration

1. **Copy the provided files into your project directory:**
   - `requirements.txt`
   - `config.yaml`
   - All Python files in `src/` directory

2. **Create directory structure:**
```bash
mkdir -p data/raw_documents data/processed data/vector_db
mkdir -p src notebooks
```

### Step 3: Document Preparation

1. **Place your documents** in `data/raw_documents/`:
   - Insurance policies (PDF)
   - Contracts (PDF, DOCX)
   - Email files (EML)
   - Text files (TXT)

2. **Example document structure:**
```
data/raw_documents/
├── health_policy_2024.pdf
├── travel_insurance.pdf
├── claims_procedure.docx
└── policy_terms.pdf
```

## 🏃 Quick Start

### Step-by-Step Execution

```bash
# Step 1: Process documents (only when new files are added)
# This performs extraction, cleaning, and chunk generation
python src/phase1_document_processing.py

# Step 2: Build semantic search index
# (embeddings are cached; add --force to regenerate)
python src/phase2_semantic_search.py [--force]
```

The processor now writes an additional
`data/processed/cleaned_text.json` file containing sanitized text that
was used for chunking.

### Other enhancements

* Queries are passed directly to the embedding model without preprocessing
  (no lowercasing or punctuation stripping) — BGE models perform better
  on natural-language text.
* Two separate instruction prefixes are now used:
  - `embeddings.instruction_prefix` — prepended to **document chunks** during indexing.
  - `embeddings.query_instruction_prefix` — prepended to **queries** during search.
  This asymmetric design is required by BGE models for accurate retrieval.
* Detailed logging has been added throughout phase‑2 (batch size,
  collection name, raw hit counts, filtered counts) to aid debugging.
* Search results include structured fields such as document name,
  chunk ID, and optional page number.

*Tip:* running the second script repeatedly without `--force` will
re-use the existing `embedded_chunks.json` and skips reinserting vectors
into ChromaDB, so you won’t keep regenerating embeddings unnecessarily.

### Use Jupyter Notebooks

Your notebooks (`Phase1.ipynb`, `Phase2.ipynb`) work with this structure. Update the file paths in them:

```python
# Update paths in your notebooks
config_path = "../config.yaml"
raw_documents_path = "../data/raw_documents/"
processed_path = "../data/processed/"
```

## 📝 Usage Examples

### Phase 1: Document Processing

```python
from src.phase1_document_processing import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()

# Process all documents
processor.process_all_documents()
```

**Expected Output:**
```
Loaded document: health_policy.pdf
Created 125 text chunks
Saved processed documents
```

### Phase 2: Semantic Search

```python
from src.phase2_semantic_search import SemanticSearch

# Initialize search engine
search = SemanticSearch()

# Build index
search.build_index()

# Test search
results = search.search("knee surgery waiting period")
```

**Expected Output:**
```
Generated embeddings for 125 chunks
Stored vectors in ChromaDB
Semantic index created

Top 5 Relevant Clauses:
1. Knee surgery requires a 24 month waiting period.
2. Orthopedic surgeries coverage conditions.
3. Joint replacement procedures terms.
```

## 🔧 Configuration

## 🖼️ Architecture Slide

For your presentation, include one simple slide showing the current
pipeline and the planned future enhancement. You can use the following
Mermaid diagram or draw it in PowerPoint/Google Slides:

```mermaid
flowchart LR
    A[Insurance Documents] --> B[Text Extraction]
    B --> C[Document Chunking]
    C --> D[Embedding Generation]
    D --> E[Vector Database (ChromaDB)]
    E --> F[Semantic Search]
    F --> G[Relevant Clauses Retrieved]

    subgraph Future
        H[LLM Reasoning
        (generate decisions)]
        G --> H
    end
```

This visual emphasizes the 50% progress and the planned LLM layer.

## 🔧 Configuration

### Key Configuration Options (config.yaml)

```yaml
# Document processing
document_processing:
  chunk_size: 200          # Words per chunk
  chunk_overlap: 50        # Overlapping words

# Embeddings
embeddings:
  model_name: "BAAI/bge-large-en-v1.5"   # upgraded from bge-base-en-v1.5
  batch_size: 32
  normalize: true
  instruction_prefix: "Represent this document for retrieval: "         # used for chunks
  query_instruction_prefix: "Represent this question for searching relevant passages: "  # used for queries

# Search settings
semantic_search:
  top_k: 5                 # Number of results to retrieve
  similarity_threshold: 0.0  # set to 0.0 to return all results (adjust as needed)
```

## 🔍 Troubleshooting

### Common Issues

1. **"Config file not found"**
   - Ensure `config.yaml` is in the project root
   - Check file permissions

2. **"No documents found"**  
   - Place PDF/DOCX files in `data/raw_documents/`
   - Check file permissions and formats

3. **"ChromaDB collection not found"**
   - Run Phase 2 to build the search index
   - Check `data/vector_db/` directory exists

4. **Memory issues with large documents**
   - Reduce `chunk_size` in config.yaml
   - Process documents in smaller batches

5. **Low similarity scores**
   - Ensure `instruction_prefix` (for chunks) and `query_instruction_prefix` (for queries)
     are set separately in `config.yaml` — using the same prefix for both degrades scores
   - Do not preprocess queries before encoding (no lowercasing/punctuation stripping)
   - Use `bge-large-en-v1.5` instead of `bge-base-en-v1.5` for better accuracy
   - After changing the model, delete the old ChromaDB collection and rebuild with `--force`:
     ```powershell
     Remove-Item -Recurse -Force data\vector_db\chroma_db
     New-Item -ItemType Directory -Path data\vector_db\chroma_db
     python src/phase2_semantic_search.py --force
     ```

6. **PyTorch not using GPU (CUDA not available)**
   - Default PyTorch install is CPU-only. Reinstall with CUDA support:
     ```bash
     pip uninstall torch torchvision torchaudio -y
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   - Verify with:
     ```bash
     python -c "import torch; print(torch.cuda.is_available())"
     ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## 📈 Performance Tips

1. **For large document collections:**
   - Use smaller chunk sizes (150-200 words)
   - Enable embedding caching
   - Use GPU for embeddings (install CUDA-enabled PyTorch, see Troubleshooting)

1. **Embedding model choice:**

   | Model | Params | Embedding Dim | VRAM needed |
   |---|---|---|---|
   | bge-small-en-v1.5 | 33M | 384 | < 1 GB |
   | bge-base-en-v1.5 | 109M | 768 | ~1 GB |
   | bge-large-en-v1.5 | 335M | 1024 | ~2 GB |

   Switch models by updating `embeddings.model_name` in `config.yaml`, then delete
   the ChromaDB collection and run `--force` to rebuild (dimension mismatch will
   cause errors if you skip this).

2. **For faster queries:**
   - Reduce `top_k` in semantic search
   - Implement response caching

## 🎯 Next Steps

The following features are planned for the next phase:

1. **Query Processing Module**
   - Natural language query interpretation
   - Entity extraction
   - Query categorization

2. **LLM Analysis Module**
   - Integration with language models
   - Decision generation logic
   - Structured JSON response formatting

3. **API Layer**
   - REST API endpoints
   - Request/response handling
   - Authentication

4. **Testing & Deployment**
   - Integration tests
   - Docker containerization
   - Cloud deployment

---

**Current Status: Phase 1-2 Complete (50% Progress)**

This system provides document processing and semantic search capabilities. The next phase will add LLM reasoning and decision generation.
