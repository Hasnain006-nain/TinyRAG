from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_DIR / "results"
PAPER_DIR = PROJECT_DIR / "paper"
TABLES_DIR = PAPER_DIR / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

RETRIEVAL_FILE = (
    RESULTS_DIR / "retrieval" / "retrieval_summary.csv"
)

RAG_FILE = (
    RESULTS_DIR / "rag" / "rag_1024_top1_500.csv"
)

BASELINE_FILE = (
    RESULTS_DIR / "baseline" / "baseline_500.csv"
)

COMPARISON_FILE = (
    RESULTS_DIR
    / "comparison"
    / "rag_vs_baseline_summary.csv"
)

BOOTSTRAP_FILE = (
    RESULTS_DIR
    / "comparison"
    / "bootstrap_results.csv"
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n{path}"
        )


def mean(df, column):
    return float(
        pd.to_numeric(
            df[column],
            errors="coerce"
        ).mean()
    )


def median(df, column):
    return float(
        pd.to_numeric(
            df[column],
            errors="coerce"
        ).median()
    )


def bootstrap_row(df, metric):
    rows = df[df["metric"] == metric]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one bootstrap row for {metric}, "
            f"found {len(rows)}."
        )
    return rows.iloc[0]


# ============================================================
# MAIN
# ============================================================

def main():

    input_files = [
        RETRIEVAL_FILE,
        RAG_FILE,
        BASELINE_FILE,
        COMPARISON_FILE,
        BOOTSTRAP_FILE,
    ]

    for path in input_files:
        require_file(path)

    retrieval = pd.read_csv(RETRIEVAL_FILE)
    rag = pd.read_csv(RAG_FILE)
    baseline = pd.read_csv(BASELINE_FILE)
    comparison = pd.read_csv(COMPARISON_FILE)
    bootstrap = pd.read_csv(BOOTSTRAP_FILE)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if len(retrieval) != 3:
        raise ValueError(
            f"Expected 3 retrieval rows, found {len(retrieval)}."
        )

    if len(rag) != 500:
        raise ValueError(
            f"Expected 500 RAG rows, found {len(rag)}."
        )

    if len(baseline) != 500:
        raise ValueError(
            f"Expected 500 baseline rows, found {len(baseline)}."
        )

    # --------------------------------------------------------
    # Table 1: Retrieval
    # --------------------------------------------------------

    table1 = pd.DataFrame({
        "Chunk size": retrieval["chunk_size"].astype(int),
        "Corpus chunks": retrieval["num_chunks"].astype(int),
        "Recall@1 (%)": retrieval["recall_at_1"] * 100.0,
        "Recall@5 (%)": retrieval["recall_at_5"] * 100.0,
        "Recall@10 (%)": retrieval["recall_at_10"] * 100.0,
        "MRR": retrieval["mrr"],
        "Mean retrieval (ms)": retrieval["mean_retrieval_time_ms"],
        "P95 retrieval (ms)": retrieval["p95_retrieval_time_ms"],
    })

    table1 = table1.round({
        "Recall@1 (%)": 1,
        "Recall@5 (%)": 1,
        "Recall@10 (%)": 1,
        "MRR": 4,
        "Mean retrieval (ms)": 3,
        "P95 retrieval (ms)": 3,
    })

    table1.to_csv(
        TABLES_DIR / "table1_retrieval_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # Table 2: End-to-end QA
    # --------------------------------------------------------

    rag_em = mean(rag, "exact_match") * 100.0
    rag_f1 = mean(rag, "f1")
    rag_gen_mean = mean(rag, "generation_time_sec")
    rag_gen_median = median(rag, "generation_time_sec")
    rag_ram = mean(rag, "available_ram_gb")

    baseline_em = mean(baseline, "exact_match") * 100.0
    baseline_f1 = mean(baseline, "f1")
    baseline_gen_mean = mean(
        baseline,
        "generation_time_sec"
    )
    baseline_gen_median = median(
        baseline,
        "generation_time_sec"
    )
    baseline_ram = mean(
        baseline,
        "available_ram_gb"
    )

    table2 = pd.DataFrame([
        {
            "System": "No-RAG baseline",
            "Context": "None",
            "Exact Match (%)": baseline_em,
            "Token F1": baseline_f1,
            "Mean generation (s)": baseline_gen_mean,
            "Median generation (s)": baseline_gen_median,
            "Mean available RAM (GB)": baseline_ram,
        },
        {
            "System": "TinyRAG",
            "Context": "1024-char Top-1",
            "Exact Match (%)": rag_em,
            "Token F1": rag_f1,
            "Mean generation (s)": rag_gen_mean,
            "Median generation (s)": rag_gen_median,
            "Mean available RAM (GB)": rag_ram,
        },
    ])

    table2 = table2.round({
        "Exact Match (%)": 2,
        "Token F1": 4,
        "Mean generation (s)": 2,
        "Median generation (s)": 2,
        "Mean available RAM (GB)": 2,
    })

    table2.to_csv(
        TABLES_DIR / "table2_end_to_end_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # Table 3: Statistical comparison
    # --------------------------------------------------------

    em_bootstrap = bootstrap_row(
        bootstrap,
        "Exact Match"
    )

    f1_bootstrap = bootstrap_row(
        bootstrap,
        "Token F1"
    )

    # McNemar exact result was computed in the completed
    # paired-comparison experiment:
    # RAG correct / baseline wrong = 163
    # baseline correct / RAG wrong = 4
    # exact p < 0.0001
    table3 = pd.DataFrame([
        {
            "Metric": "Exact Match",
            "TinyRAG": rag_em / 100.0,
            "No-RAG": baseline_em / 100.0,
            "Difference": float(em_bootstrap["difference"]),
            "95% CI low": float(em_bootstrap["ci_95_low"]),
            "95% CI high": float(em_bootstrap["ci_95_high"]),
            "Paired test": "Exact McNemar",
            "p-value": "<0.0001",
        },
        {
            "Metric": "Token F1",
            "TinyRAG": rag_f1,
            "No-RAG": baseline_f1,
            "Difference": float(f1_bootstrap["difference"]),
            "95% CI low": float(f1_bootstrap["ci_95_low"]),
            "95% CI high": float(f1_bootstrap["ci_95_high"]),
            "Paired test": "Paired bootstrap",
            "p-value": "95% CI excludes 0",
        },
    ])

    table3 = table3.round({
        "TinyRAG": 4,
        "No-RAG": 4,
        "Difference": 4,
        "95% CI low": 4,
        "95% CI high": 4,
    })

    table3.to_csv(
        TABLES_DIR / "table3_statistical_comparison.csv",
        index=False
    )

    # --------------------------------------------------------
    # Markdown tables for paper drafting
    # --------------------------------------------------------

    md_parts = []

    md_parts.append("# TinyRAG Paper-Ready Tables\n")

    md_parts.append("## Table 1. Retrieval performance across chunk sizes\n")
    md_parts.append(
        table1.to_markdown(index=False)
    )

    md_parts.append("\n## Table 2. End-to-end QA performance\n")
    md_parts.append(
        table2.to_markdown(index=False)
    )

    md_parts.append(
        "\n## Table 3. Paired comparison of TinyRAG and no-RAG\n"
    )
    md_parts.append(
        table3.to_markdown(index=False)
    )

    md_parts.append(
        "\n## Statistical note\n"
        "The same 500 questions were evaluated in both systems. "
        "The exact McNemar test used 163 questions where TinyRAG "
        "was correct and the baseline was incorrect, versus 4 "
        "questions showing the reverse outcome; the exact two-sided "
        "p-value was below 0.0001. Paired bootstrap confidence "
        "intervals used 10,000 resamples with seed 42.\n"
    )

    (TABLES_DIR / "paper_tables.md").write_text(
        "\n\n".join(md_parts),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Experiment summary
    # --------------------------------------------------------

    experiment_summary = f"""# TinyRAG Experiment Summary

## Dataset and corpus

- Evaluation questions: 500
- Source benchmark: SQuAD validation subset
- Random seed: 42
- Unique source documents: 447
- Chunk overlap: 128 characters
- Chunk configurations: 512, 768, 1024 characters

## Retrieval

- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Embedding dimension: 384
- Device: CPU
- Similarity: cosine similarity using normalized embeddings
- Retrieval evaluation: Recall@1, Recall@5, Recall@10, MRR
- End-to-end retrieval configuration: 1024-character chunks, Top-1

## Generation

- Generation model: HuggingFaceTB/SmolLM2-1.7B-Instruct
- Device: CPU
- CPU threads: 4
- Maximum new tokens: 32
- Sampling: disabled
- End-to-end questions: 500

## Main retrieval results

- 512-char: Recall@1 = 72.8%, Recall@5 = 93.6%, Recall@10 = 95.2%, MRR = 0.8166
- 768-char: Recall@1 = 76.4%, Recall@5 = 94.0%, Recall@10 = 97.0%, MRR = 0.8448
- 1024-char: Recall@1 = 80.2%, Recall@5 = 95.2%, Recall@10 = 97.2%, MRR = 0.8680

## Main end-to-end results

- TinyRAG Exact Match = {rag_em:.2f}%
- TinyRAG Token F1 = {rag_f1:.4f}
- No-RAG Exact Match = {baseline_em:.2f}%
- No-RAG Token F1 = {baseline_f1:.4f}
- Exact Match difference = {rag_em - baseline_em:+.2f} percentage points
- Token F1 difference = {rag_f1 - baseline_f1:+.4f}
- TinyRAG mean generation time = {rag_gen_mean:.2f} s/question
- No-RAG mean generation time = {baseline_gen_mean:.2f} s/question

## Paired statistical analysis

- Questions paired: 500
- TinyRAG correct / baseline wrong: 163
- Baseline correct / TinyRAG wrong: 4
- Exact McNemar test: p < 0.0001
- Exact Match 95% paired-bootstrap CI for the difference: [{float(em_bootstrap['ci_95_low'])*100:.2f}, {float(em_bootstrap['ci_95_high'])*100:.2f}] percentage points
- Token F1 95% paired-bootstrap CI for the difference: [{float(f1_bootstrap['ci_95_low']):.4f}, {float(f1_bootstrap['ci_95_high']):.4f}]
- Bootstrap resamples: 10,000
- Bootstrap seed: 42

## Hardware and execution

- CPU-only experiment
- Laptop RAM: approximately 15.79 GB total
- The completed end-to-end RAG run recorded zero generation errors.
- The completed no-RAG baseline recorded zero generation errors.

## Important scope limitation

The retrieval experiment covered all three chunk sizes, but the full 500-question generation experiment was conducted using the 1024-character + Top-1 configuration. Therefore, the end-to-end results do not establish an end-to-end comparison of all three chunk sizes.
"""

    (PAPER_DIR / "experiment_summary.md").write_text(
        experiment_summary,
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("PAPER-READY TABLES COMPLETE")
    print("=" * 70)
    print(f"Table 1: {TABLES_DIR / 'table1_retrieval_results.csv'}")
    print(f"Table 2: {TABLES_DIR / 'table2_end_to_end_results.csv'}")
    print(f"Table 3: {TABLES_DIR / 'table3_statistical_comparison.csv'}")
    print(f"Markdown: {TABLES_DIR / 'paper_tables.md'}")
    print(f"Summary:  {PAPER_DIR / 'experiment_summary.md'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
