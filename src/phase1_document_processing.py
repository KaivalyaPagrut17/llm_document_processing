"""
Phase 1 - Document Processing
Extracts, cleans, and chunks text from PDF, DOCX, EML and TXT files.
"""
import json
import re
import fitz
import docx
import nltk
import yaml
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

PROJECT_ROOT = Path(__file__).parent.parent


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Failed reading {path.name}: {e}")
        return ""


class DocumentProcessor:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(PROJECT_ROOT / "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.raw_docs_path = PROJECT_ROOT / self.config["paths"]["raw_documents"]
        self.processed_path = PROJECT_ROOT / self.config["paths"]["processed_data"]
        self.processed_path.mkdir(parents=True, exist_ok=True)

        doc_cfg = self.config["document_processing"]
        self.supported_formats = set(doc_cfg["supported_formats"])
        self.chunk_method = doc_cfg.get("chunk_method", "clause")
        self.chunk_size = doc_cfg.get("chunk_size", 200)
        self.chunk_overlap = doc_cfg.get("chunk_overlap", 1)

    # ── Extraction ────────────────────────────────────────────────────────────

    @staticmethod
    def _pdf_to_pages(path: Path) -> List[Dict[str, Any]]:
        try:
            pages = []
            with fitz.open(path) as doc:
                for i, page in enumerate(doc):
                    text = page.get_text().strip()
                    if text:
                        pages.append({"page": i + 1, "text": text})
            return pages
        except Exception as e:
            logger.error(f"PDF extract failed {path.name}: {e}")
            return []

    @staticmethod
    def _docx_to_text(path: Path) -> str:
        try:
            d = docx.Document(path)
            return "\n".join(p.text for p in d.paragraphs if p.text.strip())
        except Exception as e:
            logger.error(f"DOCX extract failed {path.name}: {e}")
            return ""

    @staticmethod
    def _eml_to_text(path: Path) -> str:
        try:
            body, in_body = [], False
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if in_body:
                        body.append(line)
                    elif line.strip() == "":
                        in_body = True
            return "".join(body).strip()
        except Exception as e:
            logger.error(f"EML extract failed {path.name}: {e}")
            return ""

    @staticmethod
    def _detect_doc_type(fname: str, content_hint: str = "") -> str:
        name = fname.lower()
        text = content_hint.lower()
        if any(k in name for k in ("policy", "insurance", "insur", "coverage", "premium", "underwriting")):
            return "insurance_policy"
        if any(k in name for k in ("claim", "reimbursement", "settlement", "compensation")):
            return "claim"
        if any(k in name for k in ("contract", "agreement", "terms", "conditions")):
            return "contract"
        if any(k in name for k in ("exclusion", "exempt", "waiver")):
            return "exclusion"
        if any(k in name for k in ("schedule", "annex", "appendix", "addendum")):
            return "schedule"
        if fname.endswith(".eml"):
            return "email"
        if any(k in text for k in ("insurance", "insured", "premium", "policyholder", "coverage")):
            return "insurance_policy"
        if any(k in text for k in ("claim", "reimbursement", "settlement")):
            return "claim"
        if any(k in text for k in ("contract", "agreement", "terms and conditions")):
            return "contract"
        return "document"

    # ── Cleaning ──────────────────────────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        patterns = [
            r"Page\s+\d+\s+of\s+\d+",
            r"(?mi)^(?:CONFIDENTIAL|PROPRIETARY|INTERNAL USE ONLY).*$",
            r"(?mi)^Page\s+\d+.*$",
            r"(?m)^(From|To|Subject|Date|CC|BCC):.*$",
        ]
        for pat in patterns:
            text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE)
        lines = text.splitlines()
        if len(lines) > 1 and lines[0] == lines[1]:
            lines.pop(1)
        text = "\n".join(lines)
        text = re.sub(r"\n{2,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        return text.strip()

    # ── Chunking ──────────────────────────────────────────────────────────────

    def chunk(self, text: str) -> List[str]:
        method = self.chunk_method.lower()
        size = self.chunk_size
        overlap = self.chunk_overlap

        if method == "paragraph":
            return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        if method == "clause":
            return self._chunk_by_clause(text, size)

        sentences = nltk.sent_tokenize(text)

        if method == "hybrid":
            paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            all_chunks = []
            for p in paras:
                p_sents = nltk.sent_tokenize(p)
                current, current_words = [], 0
                for s in p_sents:
                    wc = len(s.split())
                    if current_words + wc > size and current:
                        all_chunks.append(" ".join(current).strip())
                        current = current[-overlap:]
                        current_words = sum(len(x.split()) for x in current)
                    current.append(s)
                    current_words += wc
                if current:
                    all_chunks.append(" ".join(current).strip())
            return all_chunks

        # sentence mode
        chunks, current, current_words = [], [], 0
        for s in sentences:
            wc = len(s.split())
            if current_words + wc > size and current:
                chunks.append(" ".join(current).strip())
                current = current[-overlap:]
                current_words = sum(len(x.split()) for x in current)
            current.append(s)
            current_words += wc
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    def _chunk_by_clause(self, text: str, max_words: int) -> List[str]:
        boundary = re.compile(
            r"(?m)^(?="
            r"\d+[.)\s]"
            r"|\d+\.\d+"
            r"|\([a-zA-Z]\)"
            r"|[a-zA-Z]\)"
            r"|(?:Section|SECTION|Clause|CLAUSE)\s"
            r")"
        )
        raw = [c.strip() for c in boundary.split(text) if c.strip()]
        if len(raw) <= 1:
            raw = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        chunks = []
        for clause in raw:
            if len(clause.split()) <= max_words:
                chunks.append(clause)
            else:
                sents = nltk.sent_tokenize(clause)
                current, current_words = [], 0
                for s in sents:
                    wc = len(s.split())
                    if current_words + wc > max_words and current:
                        chunks.append(" ".join(current).strip())
                        current = current[-1:]
                        current_words = sum(len(x.split()) for x in current)
                    current.append(s)
                    current_words += wc
                if current:
                    chunks.append(" ".join(current).strip())
        return [c for c in chunks if c]

    # ── Core: chunk a single file ─────────────────────────────────────────────

    def process_single_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract, clean, and chunk one file. Returns list of chunk dicts."""
        fmt = file_path.suffix.lower()
        pages, text = [], ""

        if fmt == ".pdf":
            pages = self._pdf_to_pages(file_path)
            text = "\n".join(p["text"] for p in pages)
        elif fmt == ".docx":
            text = self._docx_to_text(file_path)
        elif fmt == ".eml":
            text = self._eml_to_text(file_path)
        elif fmt == ".txt":
            text = safe_read_text(file_path)

        if not text:
            logger.warning(f"No text extracted from {file_path.name}")
            return []

        doc_type = self._detect_doc_type(file_path.name, text[:500])
        chunks, current_section, idx = [], None, 1

        segments = [(pg["page"], pg["text"]) for pg in pages] if pages else [(None, text)]

        for page_num, seg_text in segments:
            for ch in self.chunk(self.clean_text(seg_text)):
                for line in ch.splitlines():
                    if line.strip() and line.strip().upper() == line.strip() and len(line.strip()) > 3:
                        current_section = line.strip()
                        break
                meta = {
                    "file_name": file_path.name,
                    "chunk_id": f"{file_path.stem}-C{idx}",
                    "doc_type": doc_type,
                    "chunk_index": idx,
                    "word_count": len(ch.split()),
                }
                if page_num is not None:
                    meta["page"] = page_num
                if current_section:
                    meta["section_title"] = current_section
                chunks.append({**meta, "text": ch})
                idx += 1

        logger.success(f"Processed {file_path.name}: {len(chunks)} chunks")
        return chunks

    # ── Full pipeline (company docs / phase 1 CLI) ────────────────────────────

    def process_all_documents(self) -> List[Dict[str, Any]]:
        """Process every file in raw_documents/. Writes chunked_documents.json."""
        logger.info("Starting full document-processing pipeline...")
        all_chunks = []

        files = [p for p in self.raw_docs_path.iterdir() if p.suffix.lower() in self.supported_formats]
        for p in tqdm(files, desc="Processing"):
            all_chunks.extend(self.process_single_document(p))

        (self.processed_path / "chunked_documents.json").write_text(
            json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.success(f"Chunking complete: {len(all_chunks)} chunks from {len(files)} files")
        return all_chunks


def main():
    proc = DocumentProcessor()
    chunks = proc.process_all_documents()
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Output: {proc.processed_path / 'chunked_documents.json'}")


if __name__ == "__main__":
    main()
