"""Run the golden-dataset evaluation as a Langfuse Dataset Experiment.

Runs the live ReAct agent (main.AgenticRAG) against every item in the Langfuse
dataset uploaded by evaluation/langfuse_dataset.py, via `dataset.run_experiment(...)`.
For each item this:
  - creates a trace for the agent run, linked to a Langfuse "dataset run"
  - attaches every metric below as a score on that trace
  - is browsable in Langfuse under Datasets > <dataset-name> > Runs, with
    per-item and aggregate scores, and side-by-side comparison across runs

Metrics (same ones as evaluation/evaluate.py / evaluation/metrics.py):
  - accuracy_f1                      deterministic token-F1 vs the golden answer
  - retrieval_hit/precision/recall   deterministic vs golden source page(s)
  - correctness                      LLM-judged correctness vs the golden answer (0-1)
  - faithfulness                     LLM-judged groundedness in retrieved context (0-1)

Usage:
    python evaluation/langfuse_eval.py
    python evaluation/langfuse_eval.py --limit 5 --no-llm-judge   # fast, free smoke test
    python evaluation/langfuse_eval.py --run-name "gpt-4o-baseline"
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402
from langfuse import Evaluation  # noqa: E402

from main import AgenticRAG  # noqa: E402
from src.config.config import Config  # noqa: E402
from evaluation.langfuse_dataset import DEFAULT_DATASET_NAME, upload_golden_dataset  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    token_f1,
    retrieval_relevance,
    judge_correctness,
    judge_faithfulness,
)

# Tools whose output constitutes "retrieved context" the answer should be grounded in.
GROUNDING_TOOLS = {"retriever", "wikipedia", "github"}


def build_task(rag: AgenticRAG, page_tolerance: int):
    """Task function: runs the agent + deterministic retrieval scoring for one dataset item."""

    def task(*, item, **kwargs):
        question = item.input

        retrieved_docs = rag.vector_store.retrieve(question)
        retrieval = retrieval_relevance(
            retrieved_docs,
            expected_source=item.metadata["source_document"],
            expected_pages=item.metadata["expected_pages"],
            page_tolerance=page_tolerance,
        )

        nodes = rag.graph_builder.nodes
        if nodes._agent is None:
            nodes._build_agent()
        result = nodes._agent.invoke({"messages": [HumanMessage(content=question)]})
        messages = result.get("messages", [])
        answer = getattr(messages[-1], "content", "") if messages else ""

        tool_messages = [m for m in messages if isinstance(m, ToolMessage) and m.name in GROUNDING_TOOLS]
        grounding_context = "\n\n".join(m.content for m in tool_messages)

        return {
            "answer": answer or "Could not generate answer.",
            "tools_used": [m.name for m in tool_messages],
            "grounding_context": grounding_context,
            "retrieval_hit": retrieval["hit"],
            "retrieval_precision_at_k": retrieval["precision_at_k"],
            "retrieval_recall": retrieval["recall"],
        }

    return task


def accuracy_evaluator(*, output, expected_output, **kwargs):
    return Evaluation(
        name="accuracy_f1",
        value=round(token_f1(output["answer"], expected_output), 3),
    )


def retrieval_evaluator(*, output, **kwargs):
    return [
        Evaluation(name="retrieval_hit", value=bool(output["retrieval_hit"])),
        Evaluation(name="retrieval_precision_at_k", value=round(output["retrieval_precision_at_k"], 3)),
        Evaluation(name="retrieval_recall", value=round(output["retrieval_recall"], 3)),
    ]


def build_llm_judge_evaluators(llm):
    def correctness_evaluator(*, input, output, expected_output, **kwargs):
        judgment = judge_correctness(llm, input, expected_output, output["answer"])
        return Evaluation(name="correctness", value=round(judgment.score, 3), comment=judgment.reasoning)

    def faithfulness_evaluator(*, output, **kwargs):
        judgment = judge_faithfulness(
            llm,
            output["grounding_context"].strip() or "(no context was retrieved by any tool)",
            output["answer"],
        )
        return Evaluation(
            name="faithfulness",
            value=round(judgment.score, 3),
            comment=judgment.reasoning,
            metadata={"unsupported_claims": judgment.unsupported_claims},
        )

    return [correctness_evaluator, faithfulness_evaluator]


def main():
    parser = argparse.ArgumentParser(description="Run the golden-dataset evaluation as a Langfuse experiment")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--run-name", default=None, help="Name for this Langfuse dataset run")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N items")
    parser.add_argument("--page-tolerance", type=int, default=1, help="+/- pages allowed for a retrieval hit")
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip the correctness/faithfulness LLM-judge evaluators (deterministic metrics only)",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip re-uploading golden_dataset.json into Langfuse before running",
    )
    args = parser.parse_args()

    if not Config.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is not set (check your .env file).")
    if not (Config.LANGFUSE_PUBLIC_KEY and Config.LANGFUSE_SECRET_KEY):
        raise SystemExit("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set (check your .env file).")

    if not args.skip_upload:
        upload_golden_dataset(dataset_name=args.dataset_name)

    from langfuse import get_client
    langfuse = get_client()
    dataset = langfuse.get_dataset(args.dataset_name)
    if args.limit:
        dataset.items = dataset.items[: args.limit]

    print(f"Running agent over {len(dataset.items)} items from Langfuse dataset '{args.dataset_name}'")
    rag = AgenticRAG()

    evaluators = [accuracy_evaluator, retrieval_evaluator]
    if not args.no_llm_judge:
        evaluators += build_llm_judge_evaluators(rag.llm)

    result = dataset.run_experiment(
        name=args.run_name or "agentic-rag-eval",
        run_name=args.run_name,
        description="Accuracy (token-F1), retrieval relevance, and LLM-judged correctness/faithfulness",
        task=build_task(rag, args.page_tolerance),
        evaluators=evaluators,
        max_concurrency=1,  # shares one FAISS/agent instance; keep runs sequential
    )

    print(result.format(include_item_results=True))
    if result.dataset_run_url:
        print(f"\nView in Langfuse: {result.dataset_run_url}")

    langfuse.flush()


if __name__ == "__main__":
    main()
