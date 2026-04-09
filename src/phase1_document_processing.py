"""
Phase 1 – Document Processing  (UTF-8 / encoding-safe)
Extracts, cleans, chunks text from PDF, DOCX, EML and TXT files.
"""

import json
import re
import fitz                 # PyMuPDF
import docx
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm
import yaml
import mimetypes

# sentence tokenizer
import nltk

# ensure punkt is available on first run
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


# --------------------------------------------------------------------------- #
# Helper-level functions
# --------------------------------------------------------------------------- #
def safe_read_text(path: Path) -> str:
    """
    Read a plain-text file with utf-8 and graceful fallback.
    Replaces undecodable bytes so the pipeline never crashes.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.error(f"Failed reading {path.name}: {exc}")
        return ""


def is_text_file(path: Path) -> bool:
    """Naïve mime check to avoid trying to open a binary as text."""
    mime, _ = mimetypes.guess_type(path.as_posix())
    return bool(mime and mime.startswith("text"))


# --------------------------------------------------------------------------- #
class DocumentProcessor:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = self._load_config(config_path)
        self.raw_docs_path = Path(self.config["paths"]["raw_documents"])
        self.processed_path = Path(self.config["paths"]["processed_data"])
        self.processed_path.mkdir(parents=True, exist_ok=True)

        self.supported_formats = {
            ".pdf",
            ".docx",
            ".eml",
            ".txt",
        }
        doc_cfg = self.config["document_processing"]
        self.chunk_method = doc_cfg.get("chunk_method", "sentence")
        self.chunk_size = doc_cfg.get("chunk_size", 200)
        self.chunk_overlap = doc_cfg.get("chunk_overlap", 2)  # interpreted as sentences

    # --------------------------------------------------------------------- #
    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # --------------------------------------------------------------------- #
    # 1.  Scan folder → create document_log.json
    # --------------------------------------------------------------------- #
    def create_document_log(self) -> List[Dict[str, Any]]:
        if not self.raw_docs_path.exists():
            logger.error(f"Raw docs path not found: {self.raw_docs_path}")
            return []

        logger.info(f"Scanning {self.raw_docs_path}")
        log: List[Dict[str, Any]] = []

        for p in self.raw_docs_path.iterdir():
            if p.suffix.lower() not in self.supported_formats:
                continue

            entry = {
                "file_name": p.name,
                "file_path": str(p),
                "file_size": p.stat().st_size,
                "doc_type": self._detect_doc_type(p.name),
                "format": p.suffix.lower(),
            }
            log.append(entry)
            logger.debug(f"Added: {p.name}")

        (self.processed_path / "document_log.json").write_text(
            json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.success(f"Document log created with {len(log)} entries")
        return log

    @staticmethod
    def _detect_doc_type(fname: str, content_hint: str = "") -> str:
        """Detect document type from filename first, then fall back to content."""
        name = fname.lower()
        text = content_hint.lower()

        # filename-based detection
        if any(k in name for k in ("policy", "insurance", "insur", "coverage", "premium", "underwriting")):
            return "insurance_policy"
        if any(k in name for k in ("claim", "reimbursement", "settlement", "compensation")):
            return "claim"
        if any(k in name for k in ("contract", "agreement", "terms", "conditions", "tnc", "toc")):
            return "contract"
        if any(k in name for k in ("exclusion", "exempt", "waiver")):
            return "exclusion"
        if any(k in name for k in ("schedule", "annex", "appendix", "addendum")):
            return "schedule"
        if any(k in name for k in ("invoice", "receipt", "bill", "payment")):
            return "financial"
        if any(k in name for k in ("report", "summary", "statement")):
            return "report"
        if fname.endswith(".eml"):
            return "email"

        # content-based fallback using first 500 chars
        if any(k in text for k in ("insurance", "insured", "premium", "policyholder", "coverage", "sum assured")):
            return "insurance_policy"
        if any(k in text for k in ("claim", "reimbursement", "settlement")):
            return "claim"
        if any(k in text for k in ("contract", "agreement", "terms and conditions")):
            return "contract"
        if any(k in text for k in ("hr policy", "human resources", "personnel", "employee")):
            return "hr_policy"

        return "document"

    # --------------------------------------------------------------------- #
    # 2.  Extraction helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _pdf_to_text(path: Path) -> List[Dict[str, Any]]:
        """Return a list of {page, text} dicts, one per PDF page."""
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
            body: List[str] = []
            in_body = False
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

    # --------------------------------------------------------------------- #
    # 3. Extract all
    # --------------------------------------------------------------------- #
    def extract_all_text(self) -> List[Dict[str, Any]]:
        log_entries = self.create_document_log()
        extracted: List[Dict[str, Any]] = []
        logger.info("Extracting text ...")

        for entry in tqdm(log_entries, desc="Extracting"):
            p = Path(entry["file_path"])
            fmt = entry["format"]
            text = ""
            pages = []  # only populated for PDFs

            if fmt == ".pdf":
                pages = self._pdf_to_text(p)
                text = "\n".join(pg["text"] for pg in pages)
            elif fmt == ".docx":
                text = self._docx_to_text(p)
            elif fmt == ".eml":
                text = self._eml_to_text(p)
            elif fmt == ".txt" and is_text_file(p):
                text = safe_read_text(p)

            if text:
                extracted.append(
                    {
                        "file_name": entry["file_name"],
                        "doc_type": self._detect_doc_type(entry["file_name"], text[:500]),
                        "content": text,
                        "char_count": len(text),
                        "word_count": len(text.split()),
                        "pages": pages,  # list of {page, text} for PDFs; [] for others
                    }
                )
                logger.debug(f"Extracted: {entry['file_name']} ({len(text)} chars, {len(pages)} pages)")
            else:
                logger.warning(f"No text extracted from {entry['file_name']}")

        (self.processed_path / "extracted_text.json").write_text(
            json.dumps(extracted, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.success(f"Text extraction complete: {len(extracted)} docs")
        return extracted

    # --------------------------------------------------------------------- #
    # 4. Cleaning helpers
    # --------------------------------------------------------------------- #
    def _default_clean_patterns(self) -> Dict[str, str]:
        # patterns used to strip common noise from policy documents
        return {
            "page_numbers": r"Page\s+\d+\s+of\s+\d+",
            "headers": r"(?mi)^(?:CONFIDENTIAL|PROPRIETARY|INTERNAL USE ONLY).*$",
            "footers": r"(?mi)^Page\s+\d+.*$",
            "email_headers": r"(?m)^(From|To|Subject|Date|CC|BCC):.*$",
            "urls": r"http[s]?://\\S+",
            "extra_spaces": r"\\s{3,}",
        }

    def clean_text(self, text: str) -> str:
        """Run a sequence of regex-based cleaning passes.

        Removes headers/footers/page numbers, collapses whitespace,
        merges broken lines, and strips repeated document titles.
        """
        patterns = self._default_clean_patterns()
        for pat in patterns.values():
            text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE)

        # remove repeated titles (naively: the first line repeated)
        lines = text.splitlines()
        if len(lines) > 1 and lines[0] == lines[1]:
            lines.pop(1)
        text = "\n".join(lines)

        # collapse multiple line breaks and normalize whitespace
        text = re.sub(r"\n{2,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        # fix broken hyphenated lines (word-\nword -> wordword)
        text = re.sub(r"-\n", "", text)
        # merge lines that are broken in the middle of sentences
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        return text.strip()

    # --------------------------------------------------------------------- #
    # 5. Chunking
    # --------------------------------------------------------------------- #
    def chunk(self, text: str) -> List[str]:
        """Create chunks based on configured strategy.

        sentence - accumulate sentences up to chunk_size words
        paragraph - one chunk per blank-line paragraph
        hybrid   - paragraph boundaries first, sentence-split long paragraphs
        clause   - split on numbered clause boundaries, sentence-split only if too long
        """
        method = self.chunk_method.lower()
        size   = self.chunk_size
        overlap = self.chunk_overlap

        if method == "paragraph":
            return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        if method == "clause":
            return self._chunk_by_clause(text, size)

        sentences = nltk.sent_tokenize(text)
        chunks: List[str] = []
        current: List[str] = []
        current_words = 0

        if method == "hybrid":
            paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            all_chunks: List[str] = []
            for p in paras:
                p_sents = nltk.sent_tokenize(p)
                current, current_words = [], 0
                for s in p_sents:
                    wcount = len(s.split())
                    if current_words + wcount > size and current:
                        all_chunks.append(" ".join(current).strip())
                        current = current[-overlap:]
                        current_words = sum(len(x.split()) for x in current)
                    current.append(s)
                    current_words += wcount
                if current:
                    all_chunks.append(" ".join(current).strip())
            return all_chunks

        # default: sentence mode
        for sent in sentences:
            wcount = len(sent.split())
            if current_words + wcount > size and current:
                chunks.append(" ".join(current).strip())
                current = current[-overlap:]
                current_words = sum(len(x.split()) for x in current)
            current.append(sent)
            current_words += wcount
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    def _chunk_by_clause(self, text: str, max_words: int) -> List[str]:
        """Split on numbered clause / lettered item boundaries.

        Keeps each clause intact. Only splits a clause further if it
        exceeds max_words, in which case sentence-chunking is applied
        to that clause alone.
        """
        # clause boundary: line starting with number/letter patterns like
        # 1. / 1.1 / (a) / a) / Section / SECTION or a blank line
        clause_boundary = re.compile(
            r"(?m)^(?="
            r"\d+[.)\s]"           # 1. or 1) or 1 
            r"|\d+\.\d+"           # 1.1
            r"|\([a-zA-Z]\)"       # (a)
            r"|[a-zA-Z]\)"         # a)
            r"|(?:Section|SECTION|Clause|CLAUSE)\s"
            r")"
        )

        # split into raw clauses
        raw_clauses = [c.strip() for c in clause_boundary.split(text) if c.strip()]

        # if no clause markers found fall back to paragraph split
        if len(raw_clauses) <= 1:
            raw_clauses = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        chunks: List[str] = []
        for clause in raw_clauses:
            if len(clause.split()) <= max_words:
                # clause fits — keep it whole
                chunks.append(clause)
            else:
                # clause too long — sentence-split it
                sents = nltk.sent_tokenize(clause)
                current: List[str] = []
                current_words = 0
                for s in sents:
                    wcount = len(s.split())
                    if current_words + wcount > max_words and current:
                        chunks.append(" ".join(current).strip())
                        current = current[-1:]  # 1 sentence overlap
                        current_words = sum(len(x.split()) for x in current)
                    current.append(s)
                    current_words += wcount
                if current:
                    chunks.append(" ".join(current).strip())

        return [c for c in chunks if c]

    # --------------------------------------------------------------------- #
    # 6. Full pipeline
    # --------------------------------------------------------------------- #
    def process_all_documents(self) -> List[Dict[str, Any]]:
        logger.info("Starting full document-processing pipeline...")

        extracted = self.extract_all_text()
        cleaned_list: List[Dict[str, Any]] = []
        chunked: List[Dict[str, Any]] = []

        for doc in tqdm(extracted, desc="Cleaning & chunking"):
            cleaned_list.append({
                "file_name": doc["file_name"],
                "doc_type": doc["doc_type"],
                "content": self.clean_text(doc["content"]),
                "char_count": len(doc["content"]),
                "word_count": len(doc["content"].split()),
            })

            current_section = None
            idx = 1

            # PDFs: chunk per page so page number is exact
            if doc.get("pages"):
                for pg in doc["pages"]:
                    cleaned_page = self.clean_text(pg["text"])
                    for ch in self.chunk(cleaned_page):
                        for line in ch.splitlines():
                            if line.strip() and line.strip().upper() == line.strip() and len(line.strip()) > 3:
                                current_section = line.strip()
                                break
                        metadata = {
                            "file_name": doc["file_name"],
                            "chunk_id": f"{Path(doc['file_name']).stem}-C{idx}",
                            "doc_type": doc["doc_type"],
                            "chunk_index": idx,
                            "word_count": len(ch.split()),
                            "page": pg["page"],  # exact page, no calculation
                        }
                        if current_section:
                            metadata["section_title"] = current_section
                        chunked.append({**metadata, "text": ch})
                        idx += 1

            # non-PDF: no page info available
            else:
                cleaned_text = self.clean_text(doc["content"])
                for ch in self.chunk(cleaned_text):
                    for line in ch.splitlines():
                        if line.strip() and line.strip().upper() == line.strip() and len(line.strip()) > 3:
                            current_section = line.strip()
                            break
                    metadata = {
                        "file_name": doc["file_name"],
                        "chunk_id": f"{Path(doc['file_name']).stem}-C{idx}",
                        "doc_type": doc["doc_type"],
                        "chunk_index": idx,
                        "word_count": len(ch.split()),
                    }
                    if current_section:
                        metadata["section_title"] = current_section
                    chunked.append({**metadata, "text": ch})
                    idx += 1

        # write cleaned intermediate file
        (self.processed_path / "cleaned_text.json").write_text(
            json.dumps(cleaned_list, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        (self.processed_path / "chunked_documents.json").write_text(
            json.dumps(chunked, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.success(f"Chunking complete: {len(chunked)} chunks")
        return chunked


# --------------------------------------------------------------------------- #
def main() -> None:
    proc = DocumentProcessor()
    chunks = proc.process_all_documents()

    print("\nProcessing Summary")
    print(f"   Total chunks generated : {len(chunks)}")
    print(f"   Output folder          : {proc.processed_path}")
    print("   Files created:")
    print("     - document_log.json") 
    print("     - extracted_text.json")
    print("     - chunked_documents.json")
    print("  Sample chunk:")
    print("    ", chunks[0]["text"] if chunks else "No chunks available")

if __name__ == "__main__":
    main()
