FROM python:3.11-slim

WORKDIR /app

# system deps for PyMuPDF and other native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# download nltk punkt tokenizer at build time
RUN python -c "import nltk; nltk.download('punkt_tab')"

COPY config.yaml .
COPY src/ ./src/

# create data dirs
RUN mkdir -p data/raw_documents data/processed data/vector_db/chroma_db

EXPOSE 8000

CMD ["python", "src/phase4_api.py"]
