"""LLM-judge quality metrics, shared by the evaluation harness (evaluation/metrics.py)
and live per-query scoring (src/observability/live_scoring.py).

  - correctness:      needs a reference/golden answer - only used by the evaluation harness
  - faithfulness:      reference-free - is the answer grounded in the retrieved context?
  - answer_relevancy:  reference-free - does the answer actually address the question asked?
"""

from typing import List

from pydantic import BaseModel, Field


class CorrectnessJudgment(BaseModel):
    score: float = Field(
        ge=0, le=1,
        description="0 = completely incorrect/irrelevant, 1 = fully correct and complete relative to the reference answer",
    )
    reasoning: str = Field(description="One or two sentences justifying the score")


class FaithfulnessJudgment(BaseModel):
    score: float = Field(
        ge=0, le=1,
        description="Fraction of factual claims in the generated answer that are supported by the given context",
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Claims in the answer not supported by the context (empty if fully faithful)",
    )
    reasoning: str = Field(description="One or two sentences justifying the score")


class AnswerRelevancyJudgment(BaseModel):
    score: float = Field(
        ge=0, le=1,
        description="How directly and completely the answer addresses the question asked, regardless of factual correctness",
    )
    reasoning: str = Field(description="One or two sentences justifying the score")


_CORRECTNESS_PROMPT = """You are grading a RAG system's answer against a reference (golden) answer.

Question:
{question}

Reference answer:
{reference_answer}

Generated answer:
{generated_answer}

Score how correct and complete the generated answer is relative to the reference answer, \
on a 0.0-1.0 scale. Minor phrasing differences or extra (non-contradictory) detail should not \
be penalized. Factual errors or missing key facts should be penalized."""


_FAITHFULNESS_PROMPT = """You are checking whether a RAG system's answer is faithful (grounded) in the \
retrieved context, i.e. not hallucinated.

Retrieved context:
{context}

Generated answer:
{generated_answer}

Break the generated answer into its factual claims and determine what fraction are directly \
supported by the retrieved context above. List any unsupported claims. An answer that says it \
cannot find the information is fully faithful (score 1.0) if the context indeed lacks that info."""


_ANSWER_RELEVANCY_PROMPT = """You are checking whether a RAG system's answer is relevant to the \
question asked - i.e. it actually addresses what was asked, independent of whether it is factually \
correct or grounded in any context.

Question:
{question}

Generated answer:
{generated_answer}

Score 0.0-1.0 how directly and completely the answer addresses the question. Evasive, off-topic, \
or incomplete answers should score low. An answer that plainly states the information could not be \
found still counts as fully relevant/on-topic (score high) if that is a direct response to the question."""


def judge_correctness(llm, question: str, reference_answer: str, generated_answer: str) -> CorrectnessJudgment:
    judge = llm.with_structured_output(CorrectnessJudgment)
    prompt = _CORRECTNESS_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
    )
    return judge.invoke(prompt)


def judge_faithfulness(llm, context: str, generated_answer: str) -> FaithfulnessJudgment:
    judge = llm.with_structured_output(FaithfulnessJudgment)
    prompt = _FAITHFULNESS_PROMPT.format(context=context, generated_answer=generated_answer)
    return judge.invoke(prompt)


def judge_answer_relevancy(llm, question: str, generated_answer: str) -> AnswerRelevancyJudgment:
    judge = llm.with_structured_output(AnswerRelevancyJudgment)
    prompt = _ANSWER_RELEVANCY_PROMPT.format(question=question, generated_answer=generated_answer)
    return judge.invoke(prompt)
