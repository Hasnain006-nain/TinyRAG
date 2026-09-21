import re
import string

import pandas as pd

from config import DATA_DIR
from logger import get_logger


logger = get_logger("Top1StrictEvaluation")


INPUT_FILE = (
    DATA_DIR.parent
    / "results"
    / "generation_top1_strict_pilot_10.csv"
)

OUTPUT_FILE = (
    DATA_DIR.parent
    / "results"
    / "generation_top1_strict_pilot_10_evaluated.csv"
)


def normalize_answer(text):

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

        return float(
            prediction_tokens
            == gold_tokens
        )

    common_tokens = (
        set(prediction_tokens)
        & set(gold_tokens)
    )

    common_count = sum(
        min(
            prediction_tokens.count(token),
            gold_tokens.count(token)
        )
        for token in common_tokens
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

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing file:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    logger.info(
        "Loaded %d Top-1 strict pilot results.",
        len(df)
    )

    df["exact_match"] = df.apply(
        lambda row:
            exact_match(
                row["generated_answer"],
                row["gold_answer"]
            ),
        axis=1
    )

    df["f1"] = df.apply(
        lambda row:
            token_f1(
                row["generated_answer"],
                row["gold_answer"]
            ),
        axis=1
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("TOP-1 STRICT GENERATION PILOT EVALUATION")
    print("=" * 60)

    print(
        f"Questions: {len(df)}"
    )

    print(
        f"Exact Match: "
        f"{df['exact_match'].mean():.4f}"
    )

    print(
        f"Token F1: "
        f"{df['f1'].mean():.4f}"
    )

    print(
        f"Mean generation time: "
        f"{df['generation_time_sec'].mean():.2f} sec"
    )

    print(
        f"Median generation time: "
        f"{df['generation_time_sec'].median():.2f} sec"
    )

    print()
    print("-" * 60)
    print("QUESTION-LEVEL RESULTS")
    print("-" * 60)

    columns = [
        "question_id",
        "gold_answer",
        "generated_answer",
        "exact_match",
        "f1",
        "generation_time_sec"
    ]

    print(
        df[columns].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()