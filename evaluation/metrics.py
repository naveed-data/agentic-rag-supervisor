"""Evaluation metrics for the Agentic RAG system.

Four metrics, two deterministic and two LLM-judged:
  - accuracy:            token-level F1 overlap between generated and golden answer (deterministic)
  - retrieval_relevance: precision/recall of retrieved chunks against the golden source page(s) (deterministic)
  - correctness:         LLM judge comparing the generated answer to the golden answer (0-1)
  - faithfulness:        LLM judge checking the generated answer is grounded in the retrieved context (0-1)
"""

import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402

from src.observability.judges import (  # noqa: E402
    CorrectnessJudgment,
    FaithfulnessJudgment,
    judge_correctness,
    judge_faithfulness,
)

__all__ = [
    "token_f1",
    "retrieval_relevance",
    "CorrectnessJudgment",
    "FaithfulnessJudgment",
    "judge_correctness",
    "judge_faithfulness",
]


# ---------------------------------------------------------------------------
# Accuracy (deterministic token-F1, SQuAD-style)
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 between a generated answer and the golden answer."""
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Retrieval relevance (deterministic, page/source overlap)
# ---------------------------------------------------------------------------

def retrieval_relevance(
    retrieved_docs: List[Document],
    expected_source: str,
    expected_pages: List[int],
    page_tolerance: int = 1,
) -> dict:
    """
    Score retrieved chunks against the golden source document + page(s).

    doc.metadata["page"] from PyPDFLoader is 0-indexed; expected_pages in the
    golden dataset are 1-indexed, so we convert before comparing.
    """
    expected_pages_set = set(expected_pages)
    k = len(retrieved_docs)

    if k == 0:
        return {"hit": False, "precision_at_k": 0.0, "recall": 0.0, "k": 0}

    hits = []
    matched_pages = set()
    for doc in retrieved_docs:
        src_name = Path(doc.metadata.get("source", "")).name
        raw_page = doc.metadata.get("page")
        page_1indexed = raw_page + 1 if raw_page is not None else None

        is_hit = (
            src_name == expected_source
            and page_1indexed is not None
            and any(abs(page_1indexed - p) <= page_tolerance for p in expected_pages_set)
        )
        hits.append(is_hit)
        if is_hit and page_1indexed is not None:
            matched_pages.add(page_1indexed)

    precision_at_k = sum(hits) / k
    recall = (
        len({p for p in expected_pages_set if any(abs(p - mp) <= page_tolerance for mp in matched_pages)})
        / len(expected_pages_set)
        if expected_pages_set
        else 0.0
    )

    return {
        "hit": any(hits),
        "precision_at_k": precision_at_k,
        "recall": recall,
        "k": k,
    }


# LLM-judged metrics (correctness, faithfulness) now live in src/observability/judges.py,
# shared with live per-query scoring - imported and re-exported above.
