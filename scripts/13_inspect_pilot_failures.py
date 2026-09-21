import json
import pandas as pd

from config import DATA_DIR
from logger import get_logger


logger = get_logger("FailureInspection")


PILOT_FILE = (
    DATA_DIR.parent
    / "results"
    / "generation_pilot_10_evaluated.csv"
)

RETRIEVAL_FILE = (
    DATA_DIR.parent
    / "results"
    / "retrieval"
    / "retrieval_1024.csv"
)

CORPUS_FILE = (
    DATA_DIR
    / "corpus_chunks_1024.csv"
)


def main():

    pilot = pd.read_csv(
        PILOT_FILE,
        dtype={"question_id": str}
    )

    retrieval = pd.read_csv(
        RETRIEVAL_FILE,
        dtype={"question_id": str}
    )

    corpus = pd.read_csv(
        CORPUS_FILE
    )

    failures = pilot[
        pilot["exact_match"] == 0
    ].copy()

    print()
    print("=" * 70)
    print("PILOT FAILURE INSPECTION")
    print("=" * 70)
    print(
        f"Total pilot questions: {len(pilot)}"
    )
    print(
        f"Exact-match failures: {len(failures)}"
    )

    for _, row in failures.iterrows():

        question_id = str(
            row["question_id"]
        )

        retrieval_row = retrieval[
            retrieval["question_id"]
            == question_id
        ]

        if len(retrieval_row) != 1:
            continue

        retrieval_row = retrieval_row.iloc[0]

        top1_chunk_id = int(
            retrieval_row["top1_chunk_id"]
        )

        gold_chunk_ids = set(
            json.loads(
                retrieval_row["gold_chunk_ids"]
            )
        )

        top1_is_gold = (
            top1_chunk_id in gold_chunk_ids
        )

        context_row = corpus[
            corpus["chunk_id"]
            == top1_chunk_id
        ]

        if len(context_row) == 1:
            context = str(
                context_row.iloc[0]["text"]
            )
        else:
            context = "[CONTEXT NOT FOUND]"

        print()
        print("-" * 70)
        print(
            f"Question ID: {question_id}"
        )
        print()
        print(
            f"Question:\n{row['question']}"
        )
        print()
        print(
            f"Gold answer:\n{row['gold_answer']}"
        )
        print()
        print(
            f"Generated answer:\n"
            f"{row['generated_answer']}"
        )
        print()
        print(
            f"Top-1 chunk ID: {top1_chunk_id}"
        )
        print(
            f"Gold chunk IDs: "
            f"{sorted(gold_chunk_ids)}"
        )
        print(
            f"Top-1 is a gold chunk: "
            f"{top1_is_gold}"
        )
        print()
        print(
            "Retrieved context:"
        )
        print(
            context
        )
        print()
        print(
            f"Retrieval score: "
            f"{retrieval_row['top1_score']:.4f}"
        )
        print(
            f"F1: {row['f1']:.4f}"
        )

    print()
    print("=" * 70)
    print("END FAILURE INSPECTION")
    print("=" * 70)


if __name__ == "__main__":
    main()