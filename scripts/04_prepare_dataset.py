import random

from config import DATA_DIR, SEED
from logger import get_logger

import pandas as pd
from datasets import load_dataset


logger = get_logger("Dataset")


# ============================================================
# Settings
# ============================================================

DATASET_NAME = "rajpurkar/squad"
SPLIT = "validation"
EVAL_SIZE = 500

OUTPUT_FILE = DATA_DIR / "eval_500.csv"


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)


# ============================================================
# Main
# ============================================================

def main():

    logger.info("Starting SQuAD dataset preparation.")

    logger.info(
        "Dataset: %s | Split: %s",
        DATASET_NAME,
        SPLIT,
    )

    # --------------------------------------------------------
    # Load directly from Hugging Face
    # --------------------------------------------------------

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT,
    )

    logger.info(
        "Dataset loaded. Total examples: %d",
        len(dataset),
    )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    rows = []

    for item in dataset:

        rows.append(
            {
                "id": item["id"],
                "title": item["title"],
                "question": item["question"],
                "context": item["context"],
                "answer": item["answers"]["text"][0],
                "answer_start": item["answers"]["answer_start"][0],
            }
        )

    df = pd.DataFrame(rows)

    logger.info(
        "Converted dataset to DataFrame: %s",
        df.shape,
    )

    # --------------------------------------------------------
    # Fixed evaluation sample
    # --------------------------------------------------------

    if len(df) < EVAL_SIZE:
        raise ValueError(
            f"Dataset contains only {len(df)} rows, "
            f"but {EVAL_SIZE} are required."
        )

    eval_df = (
        df.sample(
            n=EVAL_SIZE,
            random_state=SEED,
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    eval_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    logger.info(
        "Saved fixed evaluation set to: %s",
        OUTPUT_FILE,
    )

    logger.info(
        "Evaluation examples: %d",
        len(eval_df),
    )

    print()
    print("=" * 60)
    print("SQuAD PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Full dataset: {len(df):,} examples")
    print(f"Evaluation set: {len(eval_df):,} examples")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()