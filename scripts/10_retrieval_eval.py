import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# IMPORTANT:
# Load config before importing SentenceTransformer.
from config import DATA_DIR, EMBEDDINGS_DIR, HF_CACHE_DIR
from logger import get_logger


logger = get_logger("Retrieval")


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_CONFIGS = [512, 768, 1024]

TOP_K_VALUES = [1, 5, 10]

NUM_QUESTIONS = 500


# Results directory
RESULTS_DIR = (
    DATA_DIR.parent
    / "results"
    / "retrieval"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILE PATHS
# ============================================================

QUESTION_FILE = (
    DATA_DIR / "eval_500.csv"
)

QUERY_EMBEDDINGS_FILE = (
    EMBEDDINGS_DIR
    / "query_embeddings_500.npy"
)

QUERY_IDS_FILE = (
    EMBEDDINGS_DIR
    / "query_embedding_ids_500.csv"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "retrieval_summary.csv"
)


# ============================================================
# QUERY EMBEDDINGS
# ============================================================

def load_or_create_query_embeddings(
    questions,
    model
):
    """
    Create query embeddings once and reuse them.

    Query embeddings are normalized so that:
        cosine similarity = dot product
    """

    expected_ids = (
        questions["id"]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # Reuse existing query embeddings when valid
    # --------------------------------------------------------

    if (
        QUERY_EMBEDDINGS_FILE.exists()
        and QUERY_IDS_FILE.exists()
    ):

        try:

            saved_ids = (
                pd.read_csv(
                    QUERY_IDS_FILE,
                    dtype=str
                )["id"]
                .tolist()
            )

            embeddings = np.load(
                QUERY_EMBEDDINGS_FILE,
                mmap_mode="r"
            )

            if (
                saved_ids == expected_ids
                and embeddings.shape
                == (len(questions), 384)
            ):

                logger.info(
                    "Existing query embeddings are valid. "
                    "Reusing them."
                )

                print(
                    f"Query embeddings reused: "
                    f"{embeddings.shape}"
                )

                return np.asarray(
                    embeddings,
                    dtype=np.float32
                )

        except Exception as exc:

            logger.warning(
                "Could not reuse existing query embeddings: %s",
                exc
            )

    # --------------------------------------------------------
    # Create query embeddings
    # --------------------------------------------------------

    logger.info(
        "Creating embeddings for %d questions.",
        len(questions)
    )

    texts = (
        questions["question"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    start_time = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype(
        np.float32
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        "Query embedding complete. "
        "Shape=%s, time=%.2f sec.",
        embeddings.shape,
        elapsed
    )

    # Save embeddings
    np.save(
        QUERY_EMBEDDINGS_FILE,
        embeddings
    )

    # Save corresponding IDs
    pd.DataFrame(
        {
            "id": expected_ids
        }
    ).to_csv(
        QUERY_IDS_FILE,
        index=False,
        encoding="utf-8"
    )

    logger.info(
        "Saved query embeddings to %s",
        QUERY_EMBEDDINGS_FILE
    )

    return embeddings


# ============================================================
# GOLD MAPPING
# ============================================================

def load_gold_mapping(
    chunk_size
):
    """
    Return:
        {question_id: set(gold_chunk_ids)}
    """

    gold_file = (
        DATA_DIR
        / f"gold_chunks_{chunk_size}.csv"
    )

    if not gold_file.exists():

        raise FileNotFoundError(
            f"Missing gold mapping: {gold_file}"
        )

    gold_df = pd.read_csv(
        gold_file,
        dtype={
            "question_id": str
        }
    )

    gold_map = {}

    for _, row in gold_df.iterrows():

        question_id = str(
            row["question_id"]
        )

        chunk_id = int(
            row["chunk_id"]
        )

        if question_id not in gold_map:

            gold_map[question_id] = set()

        gold_map[question_id].add(
            chunk_id
        )

    return gold_map


# ============================================================
# CHECKPOINT / RESULT FILE
# ============================================================

def get_result_file(
    chunk_size
):
    return (
        RESULTS_DIR
        / f"retrieval_{chunk_size}.csv"
    )


def get_checkpoint_file(
    chunk_size
):
    return (
        RESULTS_DIR
        / f"checkpoint_{chunk_size}.json"
    )


def load_completed_questions(
    chunk_size
):
    """
    The per-configuration CSV acts as the main
    persistent checkpoint.

    Any completed question is skipped on rerun.
    """

    result_file = get_result_file(
        chunk_size
    )

    if not result_file.exists():

        return set()

    try:

        df = pd.read_csv(
            result_file,
            dtype={
                "question_id": str
            }
        )

        if "question_id" not in df.columns:

            return set()

        return set(
            df["question_id"]
            .astype(str)
        )

    except Exception as exc:

        logger.warning(
            "Could not read existing result file "
            "%s: %s",
            result_file,
            exc
        )

        return set()


def save_checkpoint(
    chunk_size,
    completed_count,
    total_questions,
    status
):

    checkpoint_file = (
        get_checkpoint_file(
            chunk_size
        )
    )

    checkpoint = {
        "chunk_size": chunk_size,
        "completed_questions": completed_count,
        "total_questions": total_questions,
        "status": status,
        "updated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    temp_file = (
        checkpoint_file.with_suffix(
            ".tmp"
        )
    )

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            checkpoint,
            f,
            indent=2
        )

    temp_file.replace(
        checkpoint_file
    )


def append_result(
    chunk_size,
    result
):

    result_file = get_result_file(
        chunk_size
    )

    result_df = pd.DataFrame(
        [result]
    )

    file_exists = (
        result_file.exists()
    )

    result_df.to_csv(
        result_file,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8"
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    ranked_chunk_ids,
    gold_chunk_ids
):
    """
    ranked_chunk_ids:
        chunk IDs ranked from most similar to least similar.

    gold_chunk_ids:
        one or more valid answer-containing chunks.

    Returns:
        Recall@1
        Recall@5
        Recall@10
        first_gold_rank
        reciprocal_rank
    """

    ranked_chunk_ids = [
        int(x)
        for x in ranked_chunk_ids
    ]

    gold_chunk_ids = {
        int(x)
        for x in gold_chunk_ids
    }

    rank_lookup = {}

    for rank, chunk_id in enumerate(
        ranked_chunk_ids,
        start=1
    ):

        if chunk_id not in rank_lookup:

            rank_lookup[chunk_id] = rank

    gold_ranks = [
        rank_lookup[chunk_id]
        for chunk_id in gold_chunk_ids
        if chunk_id in rank_lookup
    ]

    if len(gold_ranks) == 0:

        first_gold_rank = None
        reciprocal_rank = 0.0

    else:

        first_gold_rank = min(
            gold_ranks
        )

        reciprocal_rank = (
            1.0
            / first_gold_rank
        )

    recall_at_1 = int(
        any(
            chunk_id in gold_chunk_ids
            for chunk_id
            in ranked_chunk_ids[:1]
        )
    )

    recall_at_5 = int(
        any(
            chunk_id in gold_chunk_ids
            for chunk_id
            in ranked_chunk_ids[:5]
        )
    )

    recall_at_10 = int(
        any(
            chunk_id in gold_chunk_ids
            for chunk_id
            in ranked_chunk_ids[:10]
        )
    )

    return (
        recall_at_1,
        recall_at_5,
        recall_at_10,
        first_gold_rank,
        reciprocal_rank
    )


# ============================================================
# SINGLE CONFIGURATION
# ============================================================

def evaluate_chunk_configuration(
    chunk_size,
    questions,
    query_embeddings
):

    logger.info(
        "Starting retrieval evaluation for chunk size %d.",
        chunk_size
    )

    corpus_file = (
        DATA_DIR
        / f"corpus_chunks_{chunk_size}.csv"
    )

    embedding_file = (
        EMBEDDINGS_DIR
        / f"embeddings_{chunk_size}.npy"
    )

    if not corpus_file.exists():

        raise FileNotFoundError(
            f"Missing corpus: {corpus_file}"
        )

    if not embedding_file.exists():

        raise FileNotFoundError(
            f"Missing embeddings: {embedding_file}"
        )

    chunks = pd.read_csv(
        corpus_file
    )

    corpus_embeddings = np.load(
        embedding_file,
        mmap_mode="r"
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if len(chunks) != len(
        corpus_embeddings
    ):

        raise ValueError(
            f"Chunk count mismatch for "
            f"{chunk_size}: "
            f"{len(chunks)} chunks vs "
            f"{len(corpus_embeddings)} embeddings."
        )

    if (
        corpus_embeddings.ndim != 2
        or corpus_embeddings.shape[1] != 384
    ):

        raise ValueError(
            f"Unexpected embedding shape: "
            f"{corpus_embeddings.shape}"
        )

    # Make sure chunk IDs match embedding rows
    expected_chunk_ids = np.arange(
        len(chunks)
    )

    actual_chunk_ids = (
        chunks["chunk_id"]
        .astype(int)
        .to_numpy()
    )

    if not np.array_equal(
        actual_chunk_ids,
        expected_chunk_ids
    ):

        raise ValueError(
            "chunk_id order does not match "
            "embedding row order."
        )

    gold_map = load_gold_mapping(
        chunk_size
    )

    logger.info(
        "Loaded %d questions with gold mappings.",
        len(gold_map)
    )

    # --------------------------------------------------------
    # Resume information
    # --------------------------------------------------------

    completed_questions = (
        load_completed_questions(
            chunk_size
        )
    )

    total_questions = len(
        questions
    )

    logger.info(
        "Already completed: %d/%d",
        len(completed_questions),
        total_questions
    )

    save_checkpoint(
        chunk_size,
        len(completed_questions),
        total_questions,
        "running"
    )

    # --------------------------------------------------------
    # Evaluate questions one-by-one
    # --------------------------------------------------------

    for question_index, question in (
        questions.iterrows()
    ):

        question_id = str(
            question["id"]
        )

        if question_id in (
            completed_questions
        ):

            continue

        if question_id not in gold_map:

            raise ValueError(
                f"No gold mapping found for "
                f"question {question_id}"
            )

        query_vector = (
            query_embeddings[
                question_index
            ]
        )

        # ----------------------------------------------------
        # Exact cosine similarity
        #
        # Both query and corpus embeddings are normalized,
        # therefore dot product == cosine similarity.
        # ----------------------------------------------------

        retrieval_start = (
            time.perf_counter()
        )

        scores = (
            corpus_embeddings
            @ query_vector
        )

        ranked_indices = np.argsort(
            -scores
        )

        retrieval_time_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000.0

        top_10_indices = (
            ranked_indices[:10]
        )

        top_10_chunk_ids = [
            int(
                chunks.iloc[index]["chunk_id"]
            )
            for index
            in top_10_indices
        ]

        top_10_scores = [
            float(
                scores[index]
            )
            for index
            in top_10_indices
        ]

        (
            recall_at_1,
            recall_at_5,
            recall_at_10,
            first_gold_rank,
            reciprocal_rank
        ) = calculate_metrics(
            top_10_chunk_ids,
            gold_map[question_id]
        )

        result = {
            "question_id": question_id,
            "question_index": int(
                question_index
            ),
            "chunk_size": int(
                chunk_size
            ),
            "recall_at_1": recall_at_1,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
            "first_gold_rank": (
                first_gold_rank
                if first_gold_rank is not None
                else -1
            ),
            "reciprocal_rank": (
                reciprocal_rank
            ),
            "retrieval_time_ms": (
                retrieval_time_ms
            ),
            "top1_chunk_id": (
                top_10_chunk_ids[0]
            ),
            "top1_score": (
                top_10_scores[0]
            ),
            "top10_chunk_ids": json.dumps(
                top_10_chunk_ids
            ),
            "top10_scores": json.dumps(
                top_10_scores
            ),
            "gold_chunk_ids": json.dumps(
                sorted(
                    gold_map[question_id]
                )
            )
        }

        append_result(
            chunk_size,
            result
        )

        completed_questions.add(
            question_id
        )

        save_checkpoint(
            chunk_size,
            len(completed_questions),
            total_questions,
            "running"
        )

        if (
            len(completed_questions) % 25
            == 0
            or len(completed_questions)
            == total_questions
        ):

            logger.info(
                "Chunk size %d: "
                "%d/%d questions completed.",
                chunk_size,
                len(completed_questions),
                total_questions
            )

    save_checkpoint(
        chunk_size,
        len(completed_questions),
        total_questions,
        "complete"
    )

    # --------------------------------------------------------
    # Calculate aggregate metrics
    # --------------------------------------------------------

    result_file = get_result_file(
        chunk_size
    )

    results = pd.read_csv(
        result_file
    )

    if len(results) != total_questions:

        raise ValueError(
            f"Expected {total_questions} results "
            f"but found {len(results)}."
        )

    summary = {
        "chunk_size": int(
            chunk_size
        ),
        "num_questions": int(
            len(results)
        ),
        "num_chunks": int(
            len(chunks)
        ),
        "recall_at_1": float(
            results["recall_at_1"].mean()
        ),
        "recall_at_5": float(
            results["recall_at_5"].mean()
        ),
        "recall_at_10": float(
            results["recall_at_10"].mean()
        ),
        "mrr": float(
            results["reciprocal_rank"].mean()
        ),
        "mean_first_gold_rank": float(
            results.loc[
                results["first_gold_rank"] > 0,
                "first_gold_rank"
            ].mean()
        ),
        "mean_retrieval_time_ms": float(
            results[
                "retrieval_time_ms"
            ].mean()
        ),
        "median_retrieval_time_ms": float(
            results[
                "retrieval_time_ms"
            ].median()
        ),
        "p95_retrieval_time_ms": float(
            results[
                "retrieval_time_ms"
            ].quantile(0.95)
        )
    }

    logger.info(
        "Finished chunk size %d.",
        chunk_size
    )

    print()
    print(
        f"Chunk size {chunk_size}"
    )
    print(
        "-" * 40
    )
    print(
        f"Questions:      {summary['num_questions']}"
    )
    print(
        f"Corpus chunks:  {summary['num_chunks']}"
    )
    print(
        f"Recall@1:       {summary['recall_at_1']:.4f}"
    )
    print(
        f"Recall@5:       {summary['recall_at_5']:.4f}"
    )
    print(
        f"Recall@10:      {summary['recall_at_10']:.4f}"
    )
    print(
        f"MRR:            {summary['mrr']:.4f}"
    )
    print(
        f"Mean rank:      {summary['mean_first_gold_rank']:.2f}"
    )
    print(
        f"Mean latency:   "
        f"{summary['mean_retrieval_time_ms']:.4f} ms"
    )
    print(
        f"Median latency: "
        f"{summary['median_retrieval_time_ms']:.4f} ms"
    )
    print(
        f"P95 latency:    "
        f"{summary['p95_retrieval_time_ms']:.4f} ms"
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("TinyRAG RETRIEVAL EVALUATION")
    print("=" * 60)
    print(
        f"Embedding model: {MODEL_NAME}"
    )
    print(
        "Similarity: cosine similarity "
        "(normalized dot product)"
    )
    print(
        f"Top-k: {TOP_K_VALUES}"
    )
    print(
        f"Questions: {NUM_QUESTIONS}"
    )
    print(
        "Device: CPU"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    if not QUESTION_FILE.exists():

        raise FileNotFoundError(
            f"Missing question file: "
            f"{QUESTION_FILE}"
        )

    questions = pd.read_csv(
        QUESTION_FILE,
        dtype={
            "id": str
        }
    )

    if len(questions) != NUM_QUESTIONS:

        raise ValueError(
            f"Expected {NUM_QUESTIONS} questions, "
            f"found {len(questions)}."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    logger.info(
        "Loading SentenceTransformer model."
    )

    from sentence_transformers import (
        SentenceTransformer
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(
            HF_CACHE_DIR
            / "sentence_transformers"
        )
    )

    logger.info(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # Create/reuse query embeddings
    # --------------------------------------------------------

    query_embeddings = (
        load_or_create_query_embeddings(
            questions,
            model
        )
    )

    # Free model memory before retrieval.
    del model

    # --------------------------------------------------------
    # Evaluate every chunk configuration
    # --------------------------------------------------------

    summaries = []

    for chunk_size in CHUNK_CONFIGS:

        summary = (
            evaluate_chunk_configuration(
                chunk_size,
                questions,
                query_embeddings
            )
        )

        summaries.append(
            summary
        )

    # --------------------------------------------------------
    # Save final summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 60)
    print("RETRIEVAL EVALUATION COMPLETE")
    print("=" * 60)

    print()
    print(
        summary_df[
            [
                "chunk_size",
                "num_chunks",
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "mrr",
                "mean_retrieval_time_ms",
                "p95_retrieval_time_ms"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Summary saved to:"
    )
    print(
        SUMMARY_FILE
    )

    print()
    print(
        f"Detailed results directory:"
    )
    print(
        RESULTS_DIR
    )

    print("=" * 60)


if __name__ == "__main__":
    main()