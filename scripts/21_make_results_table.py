import pandas as pd

from config import DATA_DIR
from logger import get_logger


logger = get_logger("FinalResultsTable")


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = DATA_DIR.parent

RETRIEVAL_SUMMARY = (
    PROJECT_DIR
    / "results"
    / "retrieval"
    / "retrieval_summary.csv"
)

RAG_RESULTS = (
    PROJECT_DIR
    / "results"
    / "rag"
    / "rag_1024_top1_500.csv"
)

BASELINE_RESULTS = (
    PROJECT_DIR
    / "results"
    / "baseline"
    / "baseline_500.csv"
)

COMPARISON_SUMMARY = (
    PROJECT_DIR
    / "results"
    / "comparison"
    / "rag_vs_baseline_summary.csv"
)

BOOTSTRAP_RESULTS = (
    PROJECT_DIR
    / "results"
    / "comparison"
    / "bootstrap_results.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "final"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "final_results_table.csv"
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )


def numeric_mean(df, column):

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return float(
        values.mean()
    )


def numeric_median(df, column):

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return float(
        values.median()
    )


def get_bootstrap_row(
    bootstrap_df,
    metric_name
):

    rows = bootstrap_df[
        bootstrap_df["metric"] == metric_name
    ]

    if len(rows) != 1:

        raise ValueError(
            f"Expected exactly one bootstrap row "
            f"for '{metric_name}', "
            f"found {len(rows)}."
        )

    return rows.iloc[0]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("TinyRAG FINAL RESULTS TABLE")
    print("=" * 75)

    # --------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------

    require_file(
        RETRIEVAL_SUMMARY
    )

    require_file(
        RAG_RESULTS
    )

    require_file(
        BASELINE_RESULTS
    )

    require_file(
        COMPARISON_SUMMARY
    )

    require_file(
        BOOTSTRAP_RESULTS
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    retrieval = pd.read_csv(
        RETRIEVAL_SUMMARY
    )

    rag = pd.read_csv(
        RAG_RESULTS
    )

    baseline = pd.read_csv(
        BASELINE_RESULTS
    )

    comparison_summary = pd.read_csv(
        COMPARISON_SUMMARY
    )

    bootstrap = pd.read_csv(
        BOOTSTRAP_RESULTS
    )

    logger.info(
        "Loaded retrieval summary: %d rows.",
        len(retrieval)
    )

    logger.info(
        "Loaded RAG results: %d rows.",
        len(rag)
    )

    logger.info(
        "Loaded baseline results: %d rows.",
        len(baseline)
    )

    logger.info(
        "Loaded comparison summary: %d rows.",
        len(comparison_summary)
    )

    # --------------------------------------------------------
    # Validate row counts
    # --------------------------------------------------------

    if len(retrieval) != 3:

        raise ValueError(
            f"Expected 3 retrieval configurations, "
            f"found {len(retrieval)}."
        )

    if len(rag) != 500:

        raise ValueError(
            f"Expected 500 RAG rows, "
            f"found {len(rag)}."
        )

    if len(baseline) != 500:

        raise ValueError(
            f"Expected 500 baseline rows, "
            f"found {len(baseline)}."
        )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    retrieval_columns = [
        "chunk_size",
        "num_chunks",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "mean_retrieval_time_ms",
        "median_retrieval_time_ms",
        "p95_retrieval_time_ms"
    ]

    for column in retrieval_columns:

        if column not in retrieval.columns:

            raise ValueError(
                f"Missing retrieval column: {column}"
            )

    qa_columns = [
        "exact_match",
        "f1",
        "generation_time_sec",
        "available_ram_gb",
        "generation_error"
    ]

    for column in qa_columns:

        if column not in rag.columns:

            raise ValueError(
                f"Missing RAG column: {column}"
            )

        if column not in baseline.columns:

            raise ValueError(
                f"Missing baseline column: {column}"
            )

    # --------------------------------------------------------
    # RAG metrics
    # --------------------------------------------------------

    rag_em = numeric_mean(
        rag,
        "exact_match"
    )

    rag_f1 = numeric_mean(
        rag,
        "f1"
    )

    rag_generation_mean = numeric_mean(
        rag,
        "generation_time_sec"
    )

    rag_generation_median = numeric_median(
        rag,
        "generation_time_sec"
    )

    rag_ram_mean = numeric_mean(
        rag,
        "available_ram_gb"
    )

    rag_errors = int(
        rag["generation_error"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    # --------------------------------------------------------
    # Baseline metrics
    # --------------------------------------------------------

    baseline_em = numeric_mean(
        baseline,
        "exact_match"
    )

    baseline_f1 = numeric_mean(
        baseline,
        "f1"
    )

    baseline_generation_mean = numeric_mean(
        baseline,
        "generation_time_sec"
    )

    baseline_generation_median = numeric_median(
        baseline,
        "generation_time_sec"
    )

    baseline_ram_mean = numeric_mean(
        baseline,
        "available_ram_gb"
    )

    baseline_errors = int(
        baseline["generation_error"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    # --------------------------------------------------------
    # Bootstrap statistics
    # --------------------------------------------------------

    em_bootstrap = get_bootstrap_row(
        bootstrap,
        "Exact Match"
    )

    f1_bootstrap = get_bootstrap_row(
        bootstrap,
        "Token F1"
    )

    em_difference = float(
        em_bootstrap["difference"]
    )

    em_ci_low = float(
        em_bootstrap["ci_95_low"]
    )

    em_ci_high = float(
        em_bootstrap["ci_95_high"]
    )

    f1_difference = float(
        f1_bootstrap["difference"]
    )

    f1_ci_low = float(
        f1_bootstrap["ci_95_low"]
    )

    f1_ci_high = float(
        f1_bootstrap["ci_95_high"]
    )

    # --------------------------------------------------------
    # Build retrieval table
    # --------------------------------------------------------

    retrieval_table = retrieval[
        retrieval_columns
    ].copy()

    retrieval_table[
        "recall_at_1_pct"
    ] = (
        retrieval_table["recall_at_1"]
        * 100.0
    )

    retrieval_table[
        "recall_at_5_pct"
    ] = (
        retrieval_table["recall_at_5"]
        * 100.0
    )

    retrieval_table[
        "recall_at_10_pct"
    ] = (
        retrieval_table["recall_at_10"]
        * 100.0
    )

    # --------------------------------------------------------
    # Build consolidated final table
    # --------------------------------------------------------

    rows = []

    # Retrieval configurations
    for _, row in retrieval_table.iterrows():

        rows.append(
            {
                "section": "Retrieval",
                "system":
                    "MiniLM semantic retrieval",
                "configuration":
                    f"{int(row['chunk_size'])}-char chunks",
                "num_questions": 500,
                "num_chunks":
                    int(row["num_chunks"]),
                "recall_at_1_pct":
                    row["recall_at_1_pct"],
                "recall_at_5_pct":
                    row["recall_at_5_pct"],
                "recall_at_10_pct":
                    row["recall_at_10_pct"],
                "mrr":
                    row["mrr"],
                "exact_match_pct": None,
                "token_f1": None,
                "mean_retrieval_latency_ms":
                    row["mean_retrieval_time_ms"],
                "p95_retrieval_latency_ms":
                    row["p95_retrieval_time_ms"],
                "mean_generation_latency_sec":
                    None,
                "median_generation_latency_sec":
                    None,
                "mean_available_ram_gb":
                    None,
                "generation_errors": None
            }
        )

    # Find 1024-character retrieval row
    rag_retrieval_row = retrieval_table[
        retrieval_table["chunk_size"] == 1024
    ]

    if len(rag_retrieval_row) != 1:

        raise ValueError(
            "Could not find exactly one "
            "1024-character retrieval row."
        )

    rag_retrieval_row = (
        rag_retrieval_row.iloc[0]
    )

    # TinyRAG
    rows.append(
        {
            "section": "End-to-end QA",
            "system": "TinyRAG",
            "configuration":
                "1024-char chunks + Top-1",
            "num_questions":
                len(rag),
            "num_chunks":
                int(
                    rag_retrieval_row[
                        "num_chunks"
                    ]
                ),
            "recall_at_1_pct":
                rag_retrieval_row[
                    "recall_at_1_pct"
                ],
            "recall_at_5_pct":
                rag_retrieval_row[
                    "recall_at_5_pct"
                ],
            "recall_at_10_pct":
                rag_retrieval_row[
                    "recall_at_10_pct"
                ],
            "mrr":
                rag_retrieval_row["mrr"],
            "exact_match_pct":
                rag_em * 100.0,
            "token_f1":
                rag_f1,
            "mean_retrieval_latency_ms":
                rag_retrieval_row[
                    "mean_retrieval_time_ms"
                ],
            "p95_retrieval_latency_ms":
                rag_retrieval_row[
                    "p95_retrieval_time_ms"
                ],
            "mean_generation_latency_sec":
                rag_generation_mean,
            "median_generation_latency_sec":
                rag_generation_median,
            "mean_available_ram_gb":
                rag_ram_mean,
            "generation_errors":
                rag_errors
        }
    )

    # No-RAG baseline
    rows.append(
        {
            "section": "End-to-end QA",
            "system": "No-RAG baseline",
            "configuration":
                "Question only",
            "num_questions":
                len(baseline),
            "num_chunks": 0,
            "recall_at_1_pct": None,
            "recall_at_5_pct": None,
            "recall_at_10_pct": None,
            "mrr": None,
            "exact_match_pct":
                baseline_em * 100.0,
            "token_f1":
                baseline_f1,
            "mean_retrieval_latency_ms": None,
            "p95_retrieval_latency_ms": None,
            "mean_generation_latency_sec":
                baseline_generation_mean,
            "median_generation_latency_sec":
                baseline_generation_median,
            "mean_available_ram_gb":
                baseline_ram_mean,
            "generation_errors":
                baseline_errors
        }
    )

    final_table = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Save final table
    # --------------------------------------------------------

    final_table.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Print retrieval results
    # --------------------------------------------------------

    print()
    print("-" * 75)
    print("RETRIEVAL RESULTS")
    print("-" * 75)

    print(
        retrieval_table[
            [
                "chunk_size",
                "num_chunks",
                "recall_at_1_pct",
                "recall_at_5_pct",
                "recall_at_10_pct",
                "mrr",
                "mean_retrieval_time_ms",
                "p95_retrieval_time_ms"
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Print end-to-end results
    # --------------------------------------------------------

    print()
    print("-" * 75)
    print("END-TO-END QA RESULTS")
    print("-" * 75)

    print(
        f"TinyRAG Exact Match:       "
        f"{rag_em * 100:.2f}%"
    )

    print(
        f"No-RAG Exact Match:        "
        f"{baseline_em * 100:.2f}%"
    )

    print(
        f"Exact Match difference:    "
        f"{em_difference * 100:+.2f} pp"
    )

    print()

    print(
        f"TinyRAG Token F1:          "
        f"{rag_f1:.4f}"
    )

    print(
        f"No-RAG Token F1:           "
        f"{baseline_f1:.4f}"
    )

    print(
        f"Token F1 difference:       "
        f"{f1_difference:+.4f}"
    )

    print()

    print(
        f"Exact Match 95% CI:        "
        f"[{em_ci_low * 100:+.2f}, "
        f"{em_ci_high * 100:+.2f}] pp"
    )

    print(
        f"Token F1 95% CI:           "
        f"[{f1_ci_low:+.4f}, "
        f"{f1_ci_high:+.4f}]"
    )

    print()

    print(
        f"TinyRAG mean generation:   "
        f"{rag_generation_mean:.2f} sec"
    )

    print(
        f"No-RAG mean generation:    "
        f"{baseline_generation_mean:.2f} sec"
    )

    print(
        f"TinyRAG median generation: "
        f"{rag_generation_median:.2f} sec"
    )

    print(
        f"No-RAG median generation:  "
        f"{baseline_generation_median:.2f} sec"
    )

    print()

    print(
        f"TinyRAG mean available RAM:"
        f" {rag_ram_mean:.2f} GB"
    )

    print(
        f"No-RAG mean available RAM: "
        f"{baseline_ram_mean:.2f} GB"
    )

    print()

    print(
        f"TinyRAG generation errors: "
        f"{rag_errors}"
    )

    print(
        f"No-RAG generation errors:  "
        f"{baseline_errors}"
    )

    # --------------------------------------------------------
    # Save statistical summary
    # --------------------------------------------------------

    statistical_summary = pd.DataFrame(
        [
            {
                "metric": "Exact Match",
                "rag": rag_em,
                "baseline": baseline_em,
                "difference": em_difference,
                "ci_95_low": em_ci_low,
                "ci_95_high": em_ci_high
            },
            {
                "metric": "Token F1",
                "rag": rag_f1,
                "baseline": baseline_f1,
                "difference": f1_difference,
                "ci_95_low": f1_ci_low,
                "ci_95_high": f1_ci_high
            },
            {
                "metric":
                    "Mean generation time (sec)",
                "rag":
                    rag_generation_mean,
                "baseline":
                    baseline_generation_mean,
                "difference":
                    (
                        rag_generation_mean
                        - baseline_generation_mean
                    ),
                "ci_95_low": None,
                "ci_95_high": None
            },
            {
                "metric":
                    "Median generation time (sec)",
                "rag":
                    rag_generation_median,
                "baseline":
                    baseline_generation_median,
                "difference":
                    (
                        rag_generation_median
                        - baseline_generation_median
                    ),
                "ci_95_low": None,
                "ci_95_high": None
            },
            {
                "metric":
                    "Mean available RAM (GB)",
                "rag":
                    rag_ram_mean,
                "baseline":
                    baseline_ram_mean,
                "difference":
                    (
                        rag_ram_mean
                        - baseline_ram_mean
                    ),
                "ci_95_low": None,
                "ci_95_high": None
            }
        ]
    )

    statistical_file = (
        OUTPUT_DIR
        / "end_to_end_statistical_summary.csv"
    )

    statistical_summary.to_csv(
        statistical_file,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final output paths
    # --------------------------------------------------------

    print()
    print("-" * 75)
    print("OUTPUT FILES")
    print("-" * 75)

    print(
        f"Final results table:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"End-to-end summary:\n"
        f"{statistical_file}"
    )

    print(
        f"Existing paired comparison:\n"
        f"{COMPARISON_SUMMARY}"
    )

    print()
    print("=" * 75)
    print("FINAL RESULTS TABLE COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()