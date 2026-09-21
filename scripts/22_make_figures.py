from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = (
    Path(__file__).resolve().parents[1]
)

RESULTS_DIR = (
    PROJECT_DIR / "results"
)

OUTPUT_DIR = (
    PROJECT_DIR / "figures" / "paper_600dpi"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

RETRIEVAL_FILE = (
    RESULTS_DIR
    / "retrieval"
    / "retrieval_summary.csv"
)

RAG_FILE = (
    RESULTS_DIR
    / "rag"
    / "rag_1024_top1_500.csv"
)

BASELINE_FILE = (
    RESULTS_DIR
    / "baseline"
    / "baseline_500.csv"
)

BOOTSTRAP_FILE = (
    RESULTS_DIR
    / "comparison"
    / "bootstrap_results.csv"
)


# ============================================================
# FIGURE SETTINGS
# ============================================================

DPI = 600

# Keep this neutral and publication-friendly.
# No custom colors are specified so matplotlib uses its
# default color cycle.
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path) -> None:
    """Stop early if an expected input file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def save_figure(
    fig,
    stem: str
) -> None:
    """Save PNG, TIFF, and PDF copies."""
    png_path = (
        OUTPUT_DIR / f"{stem}.png"
    )
    tiff_path = (
        OUTPUT_DIR / f"{stem}.tiff"
    )
    pdf_path = (
        OUTPUT_DIR / f"{stem}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        tiff_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {tiff_path}")
    print(f"Saved: {pdf_path}")


def load_data():
    """Load and validate the completed experiment results."""
    for path in (
        RETRIEVAL_FILE,
        RAG_FILE,
        BASELINE_FILE,
        BOOTSTRAP_FILE,
    ):
        require_file(path)

    retrieval = pd.read_csv(
        RETRIEVAL_FILE
    )

    rag = pd.read_csv(
        RAG_FILE
    )

    baseline = pd.read_csv(
        BASELINE_FILE
    )

    bootstrap = pd.read_csv(
        BOOTSTRAP_FILE
    )

    if len(retrieval) != 3:
        raise ValueError(
            "Expected exactly 3 retrieval configurations."
        )

    if len(rag) != 500:
        raise ValueError(
            f"Expected 500 RAG rows, found {len(rag)}."
        )

    if len(baseline) != 500:
        raise ValueError(
            f"Expected 500 baseline rows, found {len(baseline)}."
        )

    return (
        retrieval,
        rag,
        baseline,
        bootstrap
    )


# ============================================================
# FIGURE 1
# RETRIEVAL RECALL VS CHUNK SIZE
# ============================================================

def make_figure_1(retrieval):
    fig, ax = plt.subplots(
        figsize=(6.4, 4.0)
    )

    for column, label, marker in [
        (
            "recall_at_1",
            "Recall@1",
            "o"
        ),
        (
            "recall_at_5",
            "Recall@5",
            "s"
        ),
        (
            "recall_at_10",
            "Recall@10",
            "^"
        ),
    ]:
        ax.plot(
            retrieval["chunk_size"],
            retrieval[column] * 100.0,
            marker=marker,
            linewidth=1.7,
            markersize=5,
            label=label
        )

    ax.set_xlabel(
        "Chunk size (characters)"
    )

    ax.set_ylabel(
        "Recall (%)"
    )

    ax.set_xticks(
        retrieval["chunk_size"]
    )

    ax.set_ylim(
        65,
        100
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower right"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_1_Retrieval_Recall"
    )


# ============================================================
# FIGURE 2
# MRR VS CHUNK SIZE
# ============================================================

def make_figure_2(retrieval):
    fig, ax = plt.subplots(
        figsize=(6.4, 4.0)
    )

    ax.plot(
        retrieval["chunk_size"],
        retrieval["mrr"],
        marker="o",
        linewidth=1.7,
        markersize=5
    )

    ax.set_xlabel(
        "Chunk size (characters)"
    )

    ax.set_ylabel(
        "Mean Reciprocal Rank (MRR)"
    )

    ax.set_xticks(
        retrieval["chunk_size"]
    )

    ax.set_ylim(
        0.78,
        0.90
    )

    ax.grid(
        True,
        alpha=0.25
    )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_2_MRR"
    )


# ============================================================
# FIGURE 3
# END-TO-END EXACT MATCH
# ============================================================

def make_figure_3(
    rag,
    baseline
):
    systems = [
        "No-RAG baseline",
        "TinyRAG"
    ]

    values = [
        baseline["exact_match"].mean() * 100.0,
        rag["exact_match"].mean() * 100.0
    ]

    fig, ax = plt.subplots(
        figsize=(5.6, 4.0)
    )

    bars = ax.bar(
        systems,
        values,
        width=0.55
    )

    ax.set_ylabel(
        "Exact Match (%)"
    )

    ax.set_ylim(
        0,
        42
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(
        bars,
        values
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            value + 0.8,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_3_End_to_End_EM"
    )


# ============================================================
# FIGURE 4
# END-TO-END TOKEN F1
# ============================================================

def make_figure_4(
    rag,
    baseline
):
    systems = [
        "No-RAG baseline",
        "TinyRAG"
    ]

    values = [
        baseline["f1"].mean(),
        rag["f1"].mean()
    ]

    fig, ax = plt.subplots(
        figsize=(5.6, 4.0)
    )

    bars = ax.bar(
        systems,
        values,
        width=0.55
    )

    ax.set_ylabel(
        "Token-level F1"
    )

    ax.set_ylim(
        0,
        0.58
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(
        bars,
        values
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            value + 0.012,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_4_End_to_End_F1"
    )


# ============================================================
# FIGURE 5
# GENERATION LATENCY
# ============================================================

def make_figure_5(
    rag,
    baseline
):
    systems = [
        "No-RAG baseline",
        "TinyRAG"
    ]

    values = [
        baseline[
            "generation_time_sec"
        ].mean(),
        rag[
            "generation_time_sec"
        ].mean()
    ]

    fig, ax = plt.subplots(
        figsize=(5.6, 4.0)
    )

    bars = ax.bar(
        systems,
        values,
        width=0.55
    )

    ax.set_ylabel(
        "Mean generation time (s/question)"
    )

    ax.set_ylim(
        0,
        13
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25
    )

    for bar, value in zip(
        bars,
        values
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2.0,
            value + 0.25,
            f"{value:.2f} s",
            ha="center",
            va="bottom",
            fontsize=9
        )

    fig.tight_layout()

    save_figure(
        fig,
        "Figure_5_Generation_Latency"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("TinyRAG PAPER FIGURES -- 600 DPI")
    print("=" * 70)
    print(
        f"Project directory:\n{PROJECT_DIR}"
    )
    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )
    print(
        f"Output DPI: {DPI}"
    )
    print("=" * 70)

    (
        retrieval,
        rag,
        baseline,
        bootstrap
    ) = load_data()

    # bootstrap is loaded intentionally so the script validates
    # that the statistical-analysis file exists for the complete
    # paper results package.
    _ = bootstrap

    make_figure_1(
        retrieval
    )

    make_figure_2(
        retrieval
    )

    make_figure_3(
        rag,
        baseline
    )

    make_figure_4(
        rag,
        baseline
    )

    make_figure_5(
        rag,
        baseline
    )

    print()
    print("=" * 70)
    print("ALL FIVE FIGURES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
