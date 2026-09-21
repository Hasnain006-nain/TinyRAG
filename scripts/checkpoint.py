import json
from pathlib import Path


def load_completed_ids(checkpoint_file):
    """
    Read completed question IDs from a JSONL checkpoint file.
    """

    checkpoint_file = Path(checkpoint_file)

    completed_ids = set()

    if not checkpoint_file.exists():
        return completed_ids

    with open(
        checkpoint_file,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            completed_ids.add(
                str(record["id"])
            )

    return completed_ids


def save_result(
    checkpoint_file,
    record
):
    """
    Append one completed result to disk.
    """

    checkpoint_file = Path(checkpoint_file)

    checkpoint_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        checkpoint_file,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


def count_completed(checkpoint_file):
    """
    Return number of completed records.
    """

    return len(
        load_completed_ids(checkpoint_file)
    )