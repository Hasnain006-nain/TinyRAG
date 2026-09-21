import pandas as pd

from config import DATA_DIR


FILE = DATA_DIR / "eval_500.csv"


def main():

    if not FILE.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {FILE}"
        )

    df = pd.read_csv(FILE)

    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    required_columns = {
        "id",
        "title",
        "question",
        "context",
        "answer",
        "answer_start",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if len(df) != 500:
        raise ValueError(
            f"Expected 500 rows, found {len(df)}"
        )

    if df["id"].duplicated().any():
        raise ValueError(
            "Duplicate question IDs found."
        )

    print("\nFirst question:")
    print(df.iloc[0]["question"])

    print("\nFirst answer:")
    print(df.iloc[0]["answer"])

    print("\nDataset verification: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()