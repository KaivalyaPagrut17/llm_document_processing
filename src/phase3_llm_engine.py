"""
Phase 3: LLM Reasoning Engine
Query intent classification → context assembly → LLM answer → structured output
"""
import json
import os
import re
import yaml
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from sentence_transformers import SentenceTransformer

from phase2_semantic_search import SemanticSearchEngine

load_dotenv()


# ---------------------------------------------------------------------------
# Intent classifier — rule-based, no extra model needed
# ---------------------------------------------------------------------------
INTENT_PATTERNS = {
    "definition":      r"\bwhat is\b|\bdefine\b|\bmeaning of\b|\bdefination\b",
    "coverage_check":  r"\bis .+ covered\b|\bdoes .+ cover\b|\bwill .+ pay\b|\bam i covered\b",
    "exclusion_check": r"\bexclusion\b|\bnot covered\b|\bexcluded\b|\bwhat is not\b",
    "waiting_period":  r"\bwaiting period\b|\bhow long\b|\bwhen can i claim\b",
    "claim_procedure": r"\bhow to claim\b|\bclaim process\b|\bsteps to claim\b|\bfile a claim\b",
    "premium":         r"\bpremium\b|\bhow much\b|\bcost\b|\bprice\b",
    "grace_period":    r"\bgrace period\b|\bmissed payment\b|\blate payment\b",
    "hr_policy":       r"\bnotice period\b|\bresignation\b|\bleave\b|\bsalary\b|\bincrement\b|\bretirement\b|\bmaternity\b|\bpaternity\b|\bprobation\b|\ballowance\b",
}

def classify_intent(query: str) -> str:
    q = query.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, q):
            return intent
    return "general"


# ---------------------------------------------------------------------------
# Context builder — assembles retrieved chunks into a prompt-ready string
# ---------------------------------------------------------------------------
def build_context(chunks: List[Dict[str, Any]], token_limit: int = 3000) -> str:
    """Concatenate chunk texts with source labels, respecting token limit."""
    parts = []
    total_tokens = 0
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source = f"[Source {i}: {meta.get('file_name','?')}"
        if "page" in meta:
            source += f", Page {meta['page']}"
        if "section_title" in meta:
            source += f", Section: {meta['section_title']}"
        source += "]"
        block = f"{source}\n{chunk['text']}"
        # rough token estimate: 1 token ≈ 4 chars
        block_tokens = len(block) // 4
        if total_tokens + block_tokens > token_limit:
            logger.info(f"Context token limit reached at chunk {i}, stopping.")
            break
        parts.append(block)
        total_tokens += block_tokens
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an insurance document analyst. Answer the user's question strictly using the provided context clauses.

Rules:
- Answer only from the context. Do not use outside knowledge.
- If the answer is not in the context, say "Not found in the provided documents."
- Be concise and factual.
- Always mention which document/source your answer comes from."""

def build_prompt(query: str, context: str, intent: str) -> list:
    intent_hint = {
        "definition":      "The user wants a definition. Provide a clear, direct definition.",
        "coverage_check":  "The user wants to know if something is covered. State clearly: covered / not covered / conditionally covered.",
        "exclusion_check": "The user wants to know what is excluded. List the exclusions clearly.",
        "waiting_period":  "The user wants to know about waiting periods. State the exact duration and conditions.",
        "claim_procedure": "The user wants to know how to file a claim. List the steps clearly.",
        "premium":         "The user wants to know about premium or cost. Provide the relevant details.",
        "grace_period":    "The user wants to know about grace period. State the exact duration and conditions.",
        "hr_policy":       "The user wants to know about an HR policy rule. State the exact rule, duration or condition clearly.",
        "general":         "Answer the question as accurately as possible from the context.",
    }.get(intent, "Answer the question from the context.")

    user_message = f"{intent_hint}\n\nContext:\n{context}\n\nQuestion: {query}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message}
    ]


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
class LLMEngine:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.llm_cfg = self.config["llm"]
        # fallback to .env if token not set in config
        if not self.llm_cfg.get("hf_token"):
            self.llm_cfg["hf_token"] = os.getenv("HF_TOKEN", "")
        self.search_engine = SemanticSearchEngine(config_path)
        logger.info(f"LLM Engine ready | provider={self.llm_cfg['provider']} | model={self.llm_cfg['model']}")

    def _call_llm(self, messages: list) -> str:
        client = InferenceClient(model=self.llm_cfg["model"], token=self.llm_cfg["hf_token"])
        response = client.chat_completion(
            messages=messages,
            max_tokens=self.llm_cfg["max_tokens"],
            temperature=max(self.llm_cfg["temperature"], 0.01),
        )
        return response.choices[0].message.content.strip()

    def query(self, user_query: str) -> Dict[str, Any]:
        """Full pipeline: search → build context → LLM → structured output."""
        logger.info(f"Query: '{user_query}'")

        # Step 1: classify intent
        intent = classify_intent(user_query)
        logger.info(f"Intent: {intent}")

        # Step 2: semantic search
        chunks = self.search_engine.semantic_search(user_query)
        if not chunks:
            return {
                "query": user_query,
                "intent": intent,
                "answer": "No relevant documents found in the database.",
                "sources": []
            }

        # Step 3: build context
        context = build_context(chunks, self.llm_cfg["context_token_limit"])

        # Step 4: build messages and call LLM
        messages = build_prompt(user_query, context, intent)
        answer = self._call_llm(messages)
        logger.info("LLM response received.")

        # Step 5: structured output
        sources = []
        for chunk in chunks:
            meta = chunk["metadata"]
            source = {"document": meta.get("file_name", "?"), "chunk_id": meta.get("chunk_id", "?")}
            if "page" in meta:
                source["page"] = meta["page"]
            if "section_title" in meta:
                source["section"] = meta["section_title"]
            source["similarity_score"] = round(chunk["similarity_score"], 4)
            sources.append(source)

        return {
            "query":   user_query,
            "intent":  intent,
            "answer":  answer,
            "sources": sources
        }

    def interactive(self):
        """Interactive CLI for Phase 3."""
        print("\n" + "="*60)
        print(" LLM Document Q&A — Phase 3")
        print(f" Model    : {self.llm_cfg['model']} (HuggingFace)")
        print(" Type 'quit' to exit")
        print("="*60)

        while True:
            try:
                query = input("\nYour question: ").strip()
                if query.lower() in ("quit", "exit", "q"):
                    break
                if not query:
                    continue

                result = self.query(query)

                print(f"\nIntent   : {result['intent']}")
                print(f"\nAnswer   :\n{result['answer']}")
                print(f"\nSources  :")
                for s in result["sources"]:
                    line = f"  - {s['document']} | Chunk: {s['chunk_id']} | Score: {s['similarity_score']}"
                    if "page" in s:
                        line += f" | Page: {s['page']}"
                    if "section" in s:
                        line += f" | Section: {s['section']}"
                    print(line)
                print()

            except KeyboardInterrupt:
                break
            except ConnectionError as e:
                print(f"\nConnection Error: {e}")
            except Exception as e:
                logger.error(f"Error: {e}")


def main():
    engine = LLMEngine()
    engine.interactive()


if __name__ == "__main__":
    main()
