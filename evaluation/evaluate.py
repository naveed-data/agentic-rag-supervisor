"""Evaluation harness for the Agentic RAG system.

Runs every question in golden_dataset.json through the live agent
(main.AgenticRAG) and scores each answer on four metrics:

  - accuracy_f1                     deterministic token-F1 vs the golden answer
  - retrieval_hit/precision/recall  deterministic vs golden source page(s)
  - correctness                     LLM-judged correctness vs the golden answer (0-1)
  - faithfulness                    LLM-judged groundedness in the retrieved context (0-1)

If Langfuse is configured (.env), every item's trace also gets these 6 metrics attached
as scores directly on the trace - visible in the Langfuse Tracing view, no separate
Datasets/Experiments flow needed (see evaluation/langfuse_eval.py for that instead).

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --limit 5 --no-llm-judge   # fast, free smoke test
    python evaluation/evaluate.py --output evaluation/results/run1.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402
from langfuse import get_client  # noqa: E402

from main import AgenticRAG  # noqa: E402
from src.config.config import Config  # noqa: E402
from src.observability.langfuse_client import get_langfuse_handler, with_langfuse  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    token_f1,
    retrieval_relevance,
    judge_correctness,
    judge_faithfulness,
)

# Tools whose output constitutes "retrieved context" the answer should be grounded in.
GROUNDING_TOOLS = {"retriever", "wikipedia", "github"}


def load_dataset(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["items"]


def run_agent(rag: AgenticRAG, question: str) -> tuple:
    """
    Invoke the ReAct responder agent directly (bypassing the graph's dead-end
    'retriever' node, whose output the responder never reads - see
    src/node/reactnode.py: generate_answer builds its own tool-using agent and
    ignores state.retrieved_docs). Returns (answer, grounding_context, tools_used, trace_id).
    """
    nodes = rag.graph_builder.nodes
    if nodes._agent is None:
        nodes._build_agent()

    handler = get_langfuse_handler()
    config = with_langfuse({"run_name": "evaluation", "tags": ["evaluation"], "metadata": {"question": question}})
    result = nodes._agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    trace_id = handler.last_trace_id if handler is not None else None
    messages = result.get("messages", [])

    answer = ""
    if messages:
        answer = getattr(messages[-1], "content", "") or ""

    tool_messages = [m for m in messages if isinstance(m, ToolMessage) and m.name in GROUNDING_TOOLS]
    tools_used = [m.name for m in tool_messages]
    context = "\n\n".join(m.content for m in tool_messages)

    return answer, context, tools_used, trace_id


def push_scores_to_langfuse(trace_id: Optional[str], scored: dict) -> None:
    """Attach accuracy/retrieval/correctness/faithfulness as scores directly on the
    trace for this item, so they're visible in the Langfuse Tracing view."""
    if not trace_id:
        return

    langfuse = get_client()
    langfuse.create_score(trace_id=trace_id, name="accuracy_f1", value=scored["accuracy_f1"], data_type="NUMERIC")

    retrieval = scored["retrieval_relevance"]
    langfuse.create_score(trace_id=trace_id, name="retrieval_hit", value=bool(retrieval["hit"]), data_type="BOOLEAN")
    langfuse.create_score(
        trace_id=trace_id, name="retrieval_precision_at_k", value=retrieval["precision_at_k"], data_type="NUMERIC"
    )
    langfuse.create_score(trace_id=trace_id, name="retrieval_recall", value=retrieval["recall"], data_type="NUMERIC")

    if "correctness" in scored:
        langfuse.create_score(
            trace_id=trace_id,
            name="correctness",
            value=scored["correctness"]["score"],
            data_type="NUMERIC",
            comment=scored["correctness"]["reasoning"],
        )
    if "faithfulness" in scored:
        langfuse.create_score(
            trace_id=trace_id,
            name="faithfulness",
            value=scored["faithfulness"]["score"],
            data_type="NUMERIC",
            comment=scored["faithfulness"]["reasoning"],
        )


def evaluate_item(rag: AgenticRAG, item: dict, page_tolerance: int, use_llm_judge: bool) -> dict:
    # Retrieval quality is scored against the vector store directly - this measures
    # what the retriever *would* return for this question, independent of whether
    # the ReAct agent actually decides to call the retriever tool for it.
    retrieved_docs = rag.vector_store.retrieve(item["question"])
    retrieval = retrieval_relevance(
        retrieved_docs,
        expected_source=item["source_document"],
        expected_pages=item["expected_pages"],
        page_tolerance=page_tolerance,
    )

    generated_answer, grounding_context, tools_used, trace_id = run_agent(rag, item["question"])
    accuracy = token_f1(generated_answer, item["expected_answer"])

    scored = {
        "id": item["id"],
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "generated_answer": generated_answer,
        "category": item.get("category"),
        "tools_used": tools_used,
        "accuracy_f1": round(accuracy, 3),
        "retrieval_relevance": {
            "hit": retrieval["hit"],
            "precision_at_k": round(retrieval["precision_at_k"], 3),
            "recall": round(retrieval["recall"], 3),
            "k": retrieval["k"],
        },
    }

    if use_llm_judge:
        correctness = judge_correctness(
            rag.llm, item["question"], item["expected_answer"], generated_answer
        )
        scored["correctness"] = {
            "score": round(correctness.score, 3),
            "reasoning": correctness.reasoning,
        }

        faithfulness = judge_faithfulness(
            rag.llm,
            grounding_context.strip() or "(no context was retrieved by any tool)",
            generated_answer,
        )
        scored["faithfulness"] = {
            "score": round(faithfulness.score, 3),
            "unsupported_claims": faithfulness.unsupported_claims,
            "reasoning": faithfulness.reasoning,
        }

    push_scores_to_langfuse(trace_id, scored)

    return scored


def summarize(results: list, use_llm_judge: bool) -> dict:
    n = len(results)
    summary = {
        "num_questions": n,
        "accuracy_f1_mean": round(sum(r["accuracy_f1"] for r in results) / n, 3),
        "retrieval_hit_rate": round(sum(r["retrieval_relevance"]["hit"] for r in results) / n, 3),
        "retrieval_precision_at_k_mean": round(
            sum(r["retrieval_relevance"]["precision_at_k"] for r in results) / n, 3
        ),
        "retrieval_recall_mean": round(sum(r["retrieval_relevance"]["recall"] for r in results) / n, 3),
        "agent_used_retriever_rate": round(sum("retriever" in r["tools_used"] for r in results) / n, 3),
    }
    if use_llm_judge:
        summary["correctness_mean"] = round(sum(r["correctness"]["score"] for r in results) / n, 3)
        summary["faithfulness_mean"] = round(sum(r["faithfulness"]["score"] for r in results) / n, 3)
    return summary


def print_report(results: list, summary: dict, use_llm_judge: bool) -> None:
    print("\n" + "=" * 100)
    print("PER-QUESTION RESULTS")
    print("=" * 100)
    for r in results:
        line = (
            f"[{r['id']}] accuracy={r['accuracy_f1']:.2f} "
            f"retrieval_hit={r['retrieval_relevance']['hit']} tools_used={r['tools_used']}"
        )
        if use_llm_judge:
            line += f" correctness={r['correctness']['score']:.2f} faithfulness={r['faithfulness']['score']:.2f}"
        print(line)
        print(f"   Q: {r['question']}")
        print(f"   A: {r['generated_answer'][:200]}")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for key, value in summary.items():
        print(f"{key:32s}: {value}")
    print("=" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Agentic RAG system against the golden dataset")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "evaluation" / "golden_dataset.json")
    parser.add_argument("--output", type=Path, default=None, help="Path to write the JSON report")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions")
    parser.add_argument("--page-tolerance", type=int, default=1, help="+/- pages allowed for a retrieval hit")
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip the correctness/faithfulness LLM-judge calls (deterministic metrics only)",
    )
    args = parser.parse_args()

    if not Config.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is not set (check your .env file).")

    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    print(f"Loaded {len(dataset)} golden questions from {args.dataset}")
    rag = AgenticRAG()

    use_llm_judge = not args.no_llm_judge
    results = []
    for i, item in enumerate(dataset, start=1):
        print(f"[{i}/{len(dataset)}] {item['id']}: {item['question']}")
        results.append(evaluate_item(rag, item, args.page_tolerance, use_llm_judge))

    summary = summarize(results, use_llm_judge)
    print_report(results, summary, use_llm_judge)

    output_path = args.output or (
        REPO_ROOT / "evaluation" / "results" / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"Report written to {output_path}")

    if get_langfuse_handler() is not None:
        get_client().flush()


if __name__ == "__main__":
    main()
