"""Upload evaluation/golden_dataset.json into Langfuse as a Dataset.

Idempotent: each golden item's own "id" (e.g. "attn-001") is passed as the
Langfuse dataset item id, so re-running this after editing golden_dataset.json
updates existing items in place instead of creating duplicates.

Usage:
    python evaluation/langfuse_dataset.py
    python evaluation/langfuse_dataset.py --dataset-name my-golden-set
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from langfuse import get_client  # noqa: E402
from langfuse.api.core import ApiError  # noqa: E402

from src.config import config  # noqa: E402,F401  (loads .env / Langfuse env vars as a side effect)

DEFAULT_DATASET_NAME = "agentic-rag-golden"


def load_dataset(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["items"]


def upload_golden_dataset(dataset_name: str = DEFAULT_DATASET_NAME, dataset_path: Path = None) -> None:
    dataset_path = dataset_path or (REPO_ROOT / "evaluation" / "golden_dataset.json")
    items = load_dataset(dataset_path)

    langfuse = get_client()

    try:
        langfuse.create_dataset(
            name=dataset_name,
            description="Golden QA set for the Agentic RAG system (evaluation/golden_dataset.json)",
        )
        print(f"Created Langfuse dataset '{dataset_name}'")
    except ApiError as e:
        if e.status_code != 409:
            raise
        print(f"Langfuse dataset '{dataset_name}' already exists, reusing it")

    for item in items:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            id=item["id"],
            input=item["question"],
            expected_output=item["expected_answer"],
            metadata={
                "source_document": item["source_document"],
                "expected_pages": item["expected_pages"],
                "category": item.get("category"),
            },
        )

    langfuse.flush()
    print(f"Upserted {len(items)} items into Langfuse dataset '{dataset_name}'")


def main():
    parser = argparse.ArgumentParser(description="Upload the golden dataset into Langfuse")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-path", type=Path, default=None)
    args = parser.parse_args()

    upload_golden_dataset(dataset_name=args.dataset_name, dataset_path=args.dataset_path)


if __name__ == "__main__":
    main()
