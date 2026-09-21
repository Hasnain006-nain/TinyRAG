import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from config import DATA_DIR
from logger import get_logger


logger = get_logger("RAGBaselineComparison")


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = DATA_DIR.parent

RAG_FILE = (
    PROJECT_DIR
    / "results"
    / "rag"
    / "rag_1024_top1_500.csv"
)

BASELINE_FILE = (
    PROJECT_DIR
    / "results"
    / "baseline"
    / "baseline_500.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

COMPARISON_FILE = (
    OUTPUT_DIR
    / "rag_vs_baseline_per_question.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "rag_vs_baseline_summary.csv"
)

BOOTSTRAP_FILE = (
    OUTPUT_DIR
    / "bootstrap_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
BOOTSTRAP_SAMPLES = 10000


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_mean_difference(
    rag_values,
    baseline_values,
    rng,
    n_bootstrap=BOOTSTRAP_SAMPLES
):
    """
    Paired bootstrap of mean(RAG - baseline).
    """

    rag_values = np.asarray(
        rag_values,
        dtype=np.float64
    )

    baseline_values = np.asarray(
        baseline_values,
        dtype=np.float64
    )

    differences = (
        rag_values
        - baseline_values
    )

    n = len(differences)

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=np.float64
    )

    for i in range(n_bootstrap):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        bootstrap_means[i] = (
            differences[indices].mean()
        )

    observed = differences.mean()

    lower = np.percentile(
        bootstrap_means,
        2.5
    )

    upper = np.percentile(
        bootstrap_means,
        97.5
    )

    return (
        observed,
        lower,
        upper
    )


# ============================================================
# MCNEMAR
# ============================================================

def mcnemar_exact(
    rag_em,
    baseline_em
):
    """
    Exact McNemar test using the discordant pairs.

    b = baseline correct, RAG incorrect
    c = baseline incorrect, RAG correct
    """

    rag_em = np.asarray(
        rag_em,
        dtype=int
    )

    baseline_em = np.asarray(
        baseline_em,
        dtype=int
    )

    baseline_correct_rag_wrong = int(
        np.sum(
            (baseline_em == 1)
            & (rag_em == 0)
        )
    )

    baseline_wrong_rag_correct = int(
        np.sum(
            (baseline_em == 0)
            & (rag_em == 1)
        )
    )

    b = baseline_correct_rag_wrong
    c = baseline_wrong_rag_correct

    discordant = b + c

    if discordant == 0:

        p_value = 1.0

    else:

        p_value = (
            binomtest(
                min(b, c),
                n=discordant,
                p=0.5,
                alternative="two-sided"
            )
            .pvalue
        )

    return (
        b,
        c,
        discordant,
        p_value
    )


# ============================================================
# UTILITY
# ============================================================

def safe_mean(series):

    return float(
        pd.to_numeric(
            series,
            errors="coerce"
        ).mean()
    )


def safe_median(series):

    return float(
        pd.to_numeric(
            series,
            errors="coerce"
        ).median()
    )


def percentage(value):

    return (
        float(value) * 100.0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "Starting RAG vs no-RAG paired comparison."
    )

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    if not RAG_FILE.exists():

        raise FileNotFoundError(
            f"Missing RAG results:\n{RAG_FILE}"
        )

    if not BASELINE_FILE.exists():

        raise FileNotFoundError(
            f"Missing baseline results:\n"
            f"{BASELINE_FILE}"
        )

    rag = pd.read_csv(
        RAG_FILE,
        dtype={
            "question_id": str
        }
    )

    baseline = pd.read_csv(
        BASELINE_FILE,
        dtype={
            "question_id": str
        }
    )

    logger.info(
        "RAG rows: %d",
        len(rag)
    )

    logger.info(
        "Baseline rows: %d",
        len(baseline)
    )

    # --------------------------------------------------------
    # Validate uniqueness
    # --------------------------------------------------------

    if rag["question_id"].duplicated().any():

        raise ValueError(
            "RAG results contain duplicate question IDs."
        )

    if baseline["question_id"].duplicated().any():

        raise ValueError(
            "Baseline results contain duplicate question IDs."
        )

    # --------------------------------------------------------
    # Merge paired results
    # --------------------------------------------------------

    merged = rag.merge(
        baseline,
        on="question_id",
        how="inner",
        suffixes=(
            "_rag",
            "_baseline"
        ),
        validate="one_to_one"
    )

    if len(merged) != 500:

        raise ValueError(
            f"Expected 500 paired questions, "
            f"found {len(merged)}."
        )

    logger.info(
        "Paired questions: %d",
        len(merged)
    )

    # --------------------------------------------------------
    # Validate IDs
    # --------------------------------------------------------

    rag_ids = set(
        rag["question_id"]
    )

    baseline_ids = set(
        baseline["question_id"]
    )

    if rag_ids != baseline_ids:

        missing_in_baseline = (
            rag_ids - baseline_ids
        )

        missing_in_rag = (
            baseline_ids - rag_ids
        )

        raise ValueError(
            "Question ID mismatch.\n"
            f"Missing in baseline: "
            f"{len(missing_in_baseline)}\n"
            f"Missing in RAG: "
            f"{len(missing_in_rag)}"
        )

    # --------------------------------------------------------
    # Rename useful columns explicitly
    # --------------------------------------------------------

    comparison = merged[
        [
            "question_id",
            "question_rag",
            "gold_answer_rag",
            "generated_answer_rag",
            "exact_match_rag",
            "f1_rag",
            "generation_time_sec_rag",
            "available_ram_gb_rag",

            "generated_answer_baseline",
            "exact_match_baseline",
            "f1_baseline",
            "generation_time_sec_baseline",
            "available_ram_gb_baseline"
        ]
    ].copy()

    comparison.rename(
        columns={
            "question_rag":
                "question",
            "gold_answer_rag":
                "gold_answer",
        },
        inplace=True
    )

    # --------------------------------------------------------
    # Per-question differences
    # --------------------------------------------------------

    comparison["em_difference"] = (
        comparison["exact_match_rag"]
        - comparison["exact_match_baseline"]
    )

    comparison["f1_difference"] = (
        comparison["f1_rag"]
        - comparison["f1_baseline"]
    )

    comparison["generation_time_difference_sec"] = (
        comparison["generation_time_sec_rag"]
        - comparison["generation_time_sec_baseline"]
    )

    comparison["available_ram_difference_gb"] = (
        comparison["available_ram_gb_rag"]
        - comparison["available_ram_gb_baseline"]
    )

    comparison["rag_em_wins"] = (
        comparison["em_difference"] > 0
    ).astype(int)

    comparison["baseline_em_wins"] = (
        comparison["em_difference"] < 0
    ).astype(int)

    comparison["em_tie"] = (
        comparison["em_difference"] == 0
    ).astype(int)

    comparison["rag_f1_wins"] = (
        comparison["f1_difference"] > 0
    ).astype(int)

    comparison["baseline_f1_wins"] = (
        comparison["f1_difference"] < 0
    ).astype(int)

    comparison["f1_tie"] = (
        comparison["f1_difference"] == 0
    ).astype(int)

    # --------------------------------------------------------
    # Core metrics
    # --------------------------------------------------------

    rag_em = comparison[
        "exact_match_rag"
    ].to_numpy()

    baseline_em = comparison[
        "exact_match_baseline"
    ].to_numpy()

    rag_f1 = comparison[
        "f1_rag"
    ].to_numpy()

    baseline_f1 = comparison[
        "f1_baseline"
    ].to_numpy()

    # --------------------------------------------------------
    # McNemar exact test
    # --------------------------------------------------------

    (
        baseline_correct_rag_wrong,
        baseline_wrong_rag_correct,
        discordant_pairs,
        mcnemar_p
    ) = mcnemar_exact(
        rag_em,
        baseline_em
    )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    rng = np.random.default_rng(
        SEED
    )

    (
        em_difference,
        em_ci_low,
        em_ci_high
    ) = bootstrap_mean_difference(
        rag_em,
        baseline_em,
        rng
    )

    (
        f1_difference,
        f1_ci_low,
        f1_ci_high
    ) = bootstrap_mean_difference(
        rag_f1,
        baseline_f1,
        rng
    )

    # --------------------------------------------------------
    # Save paired data
    # --------------------------------------------------------

    comparison.to_csv(
        COMPARISON_FILE,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows = [
        {
            "metric":
                "Exact Match",
            "rag":
                comparison[
                    "exact_match_rag"
                ].mean(),
            "baseline":
                comparison[
                    "exact_match_baseline"
                ].mean(),
            "difference":
                em_difference,
            "ci_95_low":
                em_ci_low,
            "ci_95_high":
                em_ci_high
        },
        {
            "metric":
                "Token F1",
            "rag":
                comparison[
                    "f1_rag"
                ].mean(),
            "baseline":
                comparison[
                    "f1_baseline"
                ].mean(),
            "difference":
                f1_difference,
            "ci_95_low":
                f1_ci_low,
            "ci_95_high":
                f1_ci_high
        },
        {
            "metric":
                "Mean generation time (sec)",
            "rag":
                safe_mean(
                    comparison[
                        "generation_time_sec_rag"
                    ]
                ),
            "baseline":
                safe_mean(
                    comparison[
                        "generation_time_sec_baseline"
                    ]
                ),
            "difference":
                safe_mean(
                    comparison[
                        "generation_time_difference_sec"
                    ]
                ),
            "ci_95_low":
                np.nan,
            "ci_95_high":
                np.nan
        },
        {
            "metric":
                "Median generation time (sec)",
            "rag":
                safe_median(
                    comparison[
                        "generation_time_sec_rag"
                    ]
                ),
            "baseline":
                safe_median(
                    comparison[
                        "generation_time_sec_baseline"
                    ]
                ),
            "difference":
                (
                    safe_median(
                        comparison[
                            "generation_time_sec_rag"
                        ]
                    )
                    -
                    safe_median(
                        comparison[
                            "generation_time_sec_baseline"
                        ]
                    )
                ),
            "ci_95_low":
                np.nan,
            "ci_95_high":
                np.nan
        },
        {
            "metric":
                "Mean available RAM (GB)",
            "rag":
                safe_mean(
                    comparison[
                        "available_ram_gb_rag"
                    ]
                ),
            "baseline":
                safe_mean(
                    comparison[
                        "available_ram_gb_baseline"
                    ]
                ),
            "difference":
                safe_mean(
                    comparison[
                        "available_ram_difference_gb"
                    ]
                ),
            "ci_95_low":
                np.nan,
            "ci_95_high":
                np.nan
        }
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Bootstrap details
    # --------------------------------------------------------

    bootstrap_df = pd.DataFrame(
        [
            {
                "metric":
                    "Exact Match",
                "difference":
                    em_difference,
                "ci_95_low":
                    em_ci_low,
                "ci_95_high":
                    em_ci_high
            },
            {
                "metric":
                    "Token F1",
                "difference":
                    f1_difference,
                "ci_95_low":
                    f1_ci_low,
                "ci_95_high":
                    f1_ci_high
            }
        ]
    )

    bootstrap_df.to_csv(
        BOOTSTRAP_FILE,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RAG vs NO-RAG PAIRED COMPARISON")
    print("=" * 70)

    print()
    print(
        f"Paired questions: {len(comparison)}"
    )

    print()
    print("-" * 70)
    print("MAIN METRICS")
    print("-" * 70)

    print(
        f"RAG Exact Match:      "
        f"{comparison['exact_match_rag'].mean():.4f} "
        f"({percentage(comparison['exact_match_rag'].mean()):.2f}%)"
    )

    print(
        f"Baseline Exact Match: "
        f"{comparison['exact_match_baseline'].mean():.4f} "
        f"({percentage(comparison['exact_match_baseline'].mean()):.2f}%)"
    )

    print(
        f"EM difference:        "
        f"{em_difference:.4f} "
        f"({percentage(em_difference):+.2f} pp)"
    )

    print(
        f"95% bootstrap CI:     "
        f"[{percentage(em_ci_low):+.2f}, "
        f"{percentage(em_ci_high):+.2f}] pp"
    )

    print()

    print(
        f"RAG Token F1:         "
        f"{comparison['f1_rag'].mean():.4f}"
    )

    print(
        f"Baseline Token F1:    "
        f"{comparison['f1_baseline'].mean():.4f}"
    )

    print(
        f"F1 difference:        "
        f"{f1_difference:+.4f}"
    )

    print(
        f"95% bootstrap CI:     "
        f"[{f1_ci_low:+.4f}, "
        f"{f1_ci_high:+.4f}]"
    )

    print()
    print("-" * 70)
    print("PAIRED EM OUTCOMES")
    print("-" * 70)

    print(
        f"RAG correct / baseline wrong: "
        f"{baseline_wrong_rag_correct}"
    )

    print(
        f"Baseline correct / RAG wrong: "
        f"{baseline_correct_rag_wrong}"
    )

    print(
        f"Both correct or both wrong: "
        f"{len(comparison) - discordant_pairs}"
    )

    print(
        f"Discordant pairs: "
        f"{discordant_pairs}"
    )

    print(
        f"McNemar exact p-value: "
        f"{mcnemar_p:.8f}"
    )

    print()
    print("-" * 70)
    print("F1 OUTCOMES")
    print("-" * 70)

    print(
        f"RAG higher F1:       "
        f"{comparison['rag_f1_wins'].sum()}"
    )

    print(
        f"Baseline higher F1:  "
        f"{comparison['baseline_f1_wins'].sum()}"
    )

    print(
        f"F1 ties:             "
        f"{comparison['f1_tie'].sum()}"
    )

    print()
    print("-" * 70)
    print("LATENCY")
    print("-" * 70)

    print(
        f"RAG mean:            "
        f"{safe_mean(comparison['generation_time_sec_rag']):.2f} sec"
    )

    print(
        f"Baseline mean:       "
        f"{safe_mean(comparison['generation_time_sec_baseline']):.2f} sec"
    )

    print(
        f"Mean difference:     "
        f"{safe_mean(comparison['generation_time_difference_sec']):+.2f} sec"
    )

    print()
    print("-" * 70)
    print("OUTPUT FILES")
    print("-" * 70)

    print(
        f"Per-question: {COMPARISON_FILE}"
    )

    print(
        f"Summary:      {SUMMARY_FILE}"
    )

    print(
        f"Bootstrap:    {BOOTSTRAP_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()