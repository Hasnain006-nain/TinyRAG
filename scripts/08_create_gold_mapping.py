import pandas as pd

from config import DATA_DIR
from logger import get_logger


logger = get_logger("GoldMapping")


QUESTION_FILE = DATA_DIR / "question_doc_map.csv"

CHUNK_CONFIGS = [512, 768, 1024]


def find_gold_chunks(question_row, chunks):

    doc_id = int(question_row["doc_id"])
    answer_start = int(question_row["answer_start"])
    answer_end = int(question_row["answer_end"])

    doc_chunks = chunks[
        chunks["doc_id"] == doc_id
    ]

    # Chunks that fully contain the complete answer span
    gold = doc_chunks[
        (doc_chunks["start"] <= answer_start)
        & (doc_chunks["end"] >= answer_end)
    ]

    return gold


def main():

    logger.info("Starting gold chunk mapping.")

    if not QUESTION_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {QUESTION_FILE}"
        )

    questions = pd.read_csv(
        QUESTION_FILE
    )

    logger.info(
        "Loaded %d questions.",
        len(questions)
    )

    for chunk_size in CHUNK_CONFIGS:

        chunk_file = (
            DATA_DIR
            / f"corpus_chunks_{chunk_size}.csv"
        )

        if not chunk_file.exists():
            raise FileNotFoundError(
                f"Missing chunk file: {chunk_file}"
            )

        chunks = pd.read_csv(
            chunk_file
        )

        records = []

        missing = 0

        for _, question in questions.iterrows():

            gold_chunks = find_gold_chunks(
                question,
                chunks
            )

            if len(gold_chunks) == 0:

                missing += 1

                logger.warning(
                    "No full-answer chunk found for question %s",
                    question["id"]
                )

                continue

            for _, chunk in gold_chunks.iterrows():

                records.append(
                    {
                        "question_id": question["id"],
                        "doc_id": int(question["doc_id"]),
                        "chunk_id": int(chunk["chunk_id"]),
                        "chunk_number": int(
                            chunk["chunk_number"]
                        ),
                        "answer_start": int(
                            question["answer_start"]
                        ),
                        "answer_end": int(
                            question["answer_end"]
                        ),
                        "answer": question["answer"],
                    }
                )

        output = pd.DataFrame(records)

        output_file = (
            DATA_DIR
            / f"gold_chunks_{chunk_size}.csv"
        )

        output.to_csv(
            output_file,
            index=False,
            encoding="utf-8"
        )

        questions_with_gold = (
            output["question_id"].nunique()
            if len(output) > 0
            else 0
        )

        logger.info(
            "Chunk size %d: %d questions have gold chunks; %d missing.",
            chunk_size,
            questions_with_gold,
            missing
        )

        logger.info(
            "Saved: %s",
            output_file
        )

        print()
        print(
            f"Chunk size {chunk_size}:"
        )
        print(
            f"  Questions: {len(questions):,}"
        )
        print(
            f"  Questions with gold chunk: "
            f"{questions_with_gold:,}"
        )
        print(
            f"  Missing gold chunk: {missing:,}"
        )
        print(
            f"  Gold mappings: {len(output):,}"
        )
        print(
            f"  Output: {output_file}"
        )

    print()
    print("=" * 60)
    print("GOLD CHUNK MAPPING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()