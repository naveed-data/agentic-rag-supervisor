"""Reference-free quality scores attached to every live app query - no golden/expected
answer needed, unlike evaluation/evaluate.py's accuracy_f1 and retrieval_hit/precision/recall,
which require a golden dataset and only run in the offline evaluation harness.

  - faithfulness:      is the answer grounded in what the tools actually retrieved?
  - answer_relevancy:  does the answer actually address the question asked?
"""

import sys

from langfuse import get_client

from src.observability.judges import judge_answer_relevancy, judge_faithfulness
from src.observability.langfuse_client import get_langfuse_handler


def score_live_trace(llm, question: str, answer: str, context: str) -> None:
    """Judge the just-completed agent run and attach scores to its Langfuse trace.
    No-ops if Langfuse isn't configured. Never raises - a scoring failure shouldn't
    break the user-facing answer."""
    handler = get_langfuse_handler()
    if handler is None:
        return
    trace_id = handler.last_trace_id
    if not trace_id:
        return

    try:
        langfuse = get_client()

        relevancy = judge_answer_relevancy(llm, question, answer)
        langfuse.create_score(
            trace_id=trace_id,
            name="answer_relevancy",
            value=round(relevancy.score, 3),
            data_type="NUMERIC",
            comment=relevancy.reasoning,
        )

        faithfulness = judge_faithfulness(llm, context.strip() or "(no context was retrieved by any tool)", answer)
        langfuse.create_score(
            trace_id=trace_id,
            name="faithfulness",
            value=round(faithfulness.score, 3),
            data_type="NUMERIC",
            comment=faithfulness.reasoning,
        )
    except Exception as e:
        print(f"[live_scoring] failed to score trace {trace_id}: {e}", file=sys.stderr)
