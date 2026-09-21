from pathlib import Path

from config import CHECKPOINTS_DIR
from checkpoint import (
    load_completed_ids,
    save_result,
)


CHECKPOINT_FILE = (
    CHECKPOINTS_DIR
    / "checkpoint_test.jsonl"
)


questions = [
    {
        "id": str(i),
        "question": f"Test question {i}"
    }
    for i in range(1, 21)
]


completed_ids = load_completed_ids(
    CHECKPOINT_FILE
)

print("=" * 60)
print("TinyRAG Checkpoint Test")
print("=" * 60)

print(
    f"Already completed: {len(completed_ids)}"
)

for question in questions:

    question_id = question["id"]

    # Skip questions already completed
    if question_id in completed_ids:
        continue

    print(
        f"Processing question {question_id}/20..."
    )

    # Simulated result
    result = {
        "id": question_id,
        "question": question["question"],
        "answer": (
            f"Test answer for question {question_id}"
        )
    }

    save_result(
        CHECKPOINT_FILE,
        result
    )

    # IMPORTANT:
    # Stop deliberately after question 10
    if question_id == "10":

        print()
        print(
            "INTENTIONAL TEST STOP AT QUESTION 10"
        )

        break


print("=" * 60)