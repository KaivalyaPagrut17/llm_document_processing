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
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


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
    def _detect_doc_type(fname: str) -> str:
        name = fname.lower()
        if any(k in name for k in ("policy", "insurance", "claim")):
            return "insurance_policy"
        if any(k in name for k in ("contract", "agreement")):
            return "contract"
        if fname.endswith(".eml"):
            return "email"
        return "document"

    # --------------------------------------------------------------------- #
    # 2.  Extraction helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _pdf_to_text(path: Path) -> str:
        try:
            text = ""
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text()
            return text.strip()
        except Exception as e:
            logger.error(f"PDF extract failed {path.name}: {e}")
            return ""

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

            if fmt == ".pdf":
                text = self._pdf_to_text(p)
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
                        "doc_type": entry["doc_type"],
                        "content": text,
                        "char_count": len(text),
                        "word_count": len(text.split()),
                    }
                )
                logger.debug(f"Extracted: {entry['file_name']} ({len(text)} chars)")
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

        sentence - split text into sentences and accumulate up to chunk_size
        paragraph - split on blank lines
        hybrid - paragraph boundaries first, then sentence-split long paragraphs
        """
        method = self.chunk_method.lower()

        if method == "paragraph":
            paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            return paras

        # sentence-based logic (also used by hybrid inside paragraphs)
        sentences = nltk.sent_tokenize(text)
        # further split sentences into clauses to avoid mid-clause cuts
        def sentence_to_clauses(sent: str) -> List[str]:
            # split on semicolons or period-space unless it's an abbreviation
            return re.split(r'(?<=[\.;])\s+', sent)
        clauses = []
        for s in sentences:
            clauses.extend(sentence_to_clauses(s))
        chunks: List[str] = []
        current: List[str] = []
        current_words = 0

        def flush():
            nonlocal current, current_words
            if current:
                chunks.append(" ".join(current).strip())
            current = []
            current_words = 0

        size = self.chunk_size
        overlap = self.chunk_overlap  # sentences

        if method == "hybrid":
            # split into paragraphs first, then apply sentence logic per paragraph
            paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            all_chunks: List[str] = []
            for p in paras:
                p_sents = nltk.sent_tokenize(p)
                # reuse sentence accumulation for each paragraph
                current = []
                current_words = 0
                for s in p_sents:
                    wcount = len(s.split())
                    if current_words + wcount > size and current:
                        all_chunks.append(" ".join(current).strip())
                        # carry overlap
                        current = current[-overlap:]
                        current_words = sum(len(x.split()) for x in current)
                    current.append(s)
                    current_words += wcount
                if current:
                    all_chunks.append(" ".join(current).strip())
            return all_chunks

        # default sentence mode
        for sent in sentences:
            wcount = len(sent.split())
            if current_words + wcount > size and current:
                # emit chunk
                chunks.append(" ".join(current).strip())
                # apply overlap in sentences
                current = current[-overlap:]
                current_words = sum(len(x.split()) for x in current)
            current.append(sent)
            current_words += wcount
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    # --------------------------------------------------------------------- #
    # 6. Full pipeline
    # --------------------------------------------------------------------- #
    def process_all_documents(self) -> List[Dict[str, Any]]:
        logger.info("Starting full document-processing pipeline...")

        extracted = self.extract_all_text()
        cleaned_list: List[Dict[str, Any]] = []
        chunked: List[Dict[str, Any]] = []

        for doc in tqdm(extracted, desc="Cleaning & chunking"):
            orig_text = doc["content"]
            cleaned_text = self.clean_text(orig_text)
            cleaned_list.append({
                "file_name": doc["file_name"],
                "doc_type": doc["doc_type"],
                "content": cleaned_text,
                "char_count": len(cleaned_text),
                "word_count": len(cleaned_text.split()),
            })

            # detect section titles (all-caps lines) in cleaned text
            section_title = None
            for line in cleaned_text.splitlines():
                if line.strip() and line.strip().upper() == line.strip():
                    section_title = line.strip()
                    break

            for idx, ch in enumerate(self.chunk(cleaned_text), 1):
                # simple clause detection: number of clauses in chunk
                metadata = {
                    "file_name": doc["file_name"],
                    "chunk_id": f"{Path(doc['file_name']).stem}-C{idx}",
                    "doc_type": doc["doc_type"],
                    "chunk_index": idx,
                    "word_count": len(ch.split()),
                }
                if section_title:
                    metadata["section_title"] = section_title
                chunked.append({**metadata, "text": ch})

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
