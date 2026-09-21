import re
import string
from pathlib import Path

import pandas as pd

from config import DATA_DIR
from logger import get_logger


logger = get_logger("PilotEvaluation")


PILOT_FILE = (
    DATA_DIR.parent
    / "results"
    / "generation_pilot_10.csv"
)

OUTPUT_FILE = (
    DATA_DIR.parent
    / "results"
    / "generation_pilot_10_evaluated.csv"
)


def normalize_answer(text):
    """
    SQuAD-style normalization:
    lowercase
    remove punctuation
    remove articles
    normalize whitespace
    """

    text = str(text).lower()

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text
    )

    text = " ".join(
        text.split()
    )

    return text


def exact_match(
    prediction,
    gold
):

    return int(
        normalize_answer(prediction)
        == normalize_answer(gold)
    )


def token_f1(
    prediction,
    gold
):

    prediction_tokens = (
        normalize_answer(prediction)
        .split()
    )

    gold_tokens = (
        normalize_answer(gold)
        .split()
    )

    if (
        len(prediction_tokens) == 0
        or len(gold_tokens) == 0
    ):

        return int(
            prediction_tokens
            == gold_tokens
        )

    common = (
        set(prediction_tokens)
        & set(gold_tokens)
    )

    common_count = sum(
        min(
            prediction_tokens.count(token),
            gold_tokens.count(token)
        )
        for token in common
    )

    if common_count == 0:
        return 0.0

    precision = (
        common_count
        / len(prediction_tokens)
    )

    recall = (
        common_count
        / len(gold_tokens)
    )

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def main():

    if not PILOT_FILE.exists():

        raise FileNotFoundError(
            f"Missing pilot file:\n"
            f"{PILOT_FILE}"
        )

    results = pd.read_csv(
        PILOT_FILE
    )

    logger.info(
        "Loaded %d pilot results.",
        len(results)
    )

    results["exact_match"] = (
        results.apply(
            lambda row:
                exact_match(
                    row["generated_answer"],
                    row["gold_answer"]
                ),
            axis=1
        )
    )

    results["f1"] = (
        results.apply(
            lambda row:
                token_f1(
                    row["generated_answer"],
                    row["gold_answer"]
                ),
            axis=1
        )
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("GENERATION PILOT EVALUATION")
    print("=" * 60)

    print(
        f"Questions: {len(results)}"
    )

    print(
        f"Exact Match: "
        f"{results['exact_match'].mean():.4f}"
    )

    print(
        f"Token F1: "
        f"{results['f1'].mean():.4f}"
    )

    print(
        f"Mean generation time: "
        f"{results['generation_time_sec'].mean():.2f} sec"
    )

    print(
        f"Median generation time: "
        f"{results['generation_time_sec'].median():.2f} sec"
    )

    print()
    print("-" * 60)
    print("QUESTION-LEVEL RESULTS")
    print("-" * 60)

    display_columns = [
        "question_id",
        "gold_answer",
        "generated_answer",
        "exact_match",
        "f1",
        "generation_time_sec"
    ]

    print(
        results[display_columns].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved evaluated results to:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 60)


if __name__ == "__main__":
    main()