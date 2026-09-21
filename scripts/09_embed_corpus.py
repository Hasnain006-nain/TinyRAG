import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# IMPORTANT:
# Load config BEFORE importing Hugging Face / SentenceTransformer.
from config import DATA_DIR, EMBEDDINGS_DIR, HF_CACHE_DIR
from logger import get_logger


logger = get_logger("Embedding")


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_CONFIGS = [512, 768, 1024]

BATCH_SIZE = 32


def get_paths(chunk_size):

    batch_dir = (
        EMBEDDINGS_DIR
        / f"chunks_{chunk_size}"
    )

    batch_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_file = (
        batch_dir / "checkpoint.json"
    )

    final_file = (
        EMBEDDINGS_DIR
        / f"embeddings_{chunk_size}.npy"
    )

    return (
        batch_dir,
        checkpoint_file,
        final_file
    )


def load_checkpoint(checkpoint_file):

    if not checkpoint_file.exists():

        return {
            "completed_batches": [],
            "total_batches": 0,
            "status": "not_started"
        }

    with open(
        checkpoint_file,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_checkpoint(
    checkpoint_file,
    completed_batches,
    total_batches,
    status
):

    checkpoint = {
        "completed_batches": completed_batches,
        "total_batches": total_batches,
        "status": status,
        "updated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    temp_file = checkpoint_file.with_suffix(
        ".tmp"
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


def consolidate_batches(
    batch_dir,
    total_batches,
    final_file
):

    arrays = []

    logger.info(
        "Consolidating %d embedding batches.",
        total_batches
    )

    for batch_number in range(
        total_batches
    ):

        batch_file = (
            batch_dir
            / f"batch_{batch_number:04d}.npy"
        )

        if not batch_file.exists():

            raise FileNotFoundError(
                f"Missing batch: {batch_file}"
            )

        arrays.append(
            np.load(
                batch_file
            )
        )

    embeddings = np.vstack(
        arrays
    ).astype(
        np.float32
    )

    np.save(
        final_file,
        embeddings
    )

    return embeddings


def embed_configuration(
    model,
    chunk_size
):

    logger.info(
        "Starting embedding for chunk size %d.",
        chunk_size
    )

    input_file = (
        DATA_DIR
        / f"corpus_chunks_{chunk_size}.csv"
    )

    if not input_file.exists():

        raise FileNotFoundError(
            f"Missing corpus file: {input_file}"
        )

    chunks = pd.read_csv(
        input_file
    )

    texts = (
        chunks["text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    total_chunks = len(texts)

    total_batches = (
        total_chunks + BATCH_SIZE - 1
    ) // BATCH_SIZE

    (
        batch_dir,
        checkpoint_file,
        final_file
    ) = get_paths(
        chunk_size
    )

    # Already completed completely
    if final_file.exists():

        embeddings = np.load(
            final_file,
            mmap_mode="r"
        )

        if (
            embeddings.shape[0]
            == total_chunks
        ):

            logger.info(
                "Final embedding file already exists "
                "for chunk size %d. Skipping.",
                chunk_size
            )

            print(
                f"Chunk size {chunk_size}: "
                f"already complete "
                f"({embeddings.shape})"
            )

            return

    checkpoint = load_checkpoint(
        checkpoint_file
    )

    completed_batches = set(
        checkpoint.get(
            "completed_batches",
            []
        )
    )

    save_checkpoint(
        checkpoint_file,
        sorted(completed_batches),
        total_batches,
        "running"
    )

    logger.info(
        "Total chunks: %d",
        total_chunks
    )

    logger.info(
        "Total batches: %d",
        total_batches
    )

    logger.info(
        "Completed batches: %d",
        len(completed_batches)
    )

    for batch_number in range(
        total_batches
    ):

        if batch_number in completed_batches:

            logger.info(
                "Skipping completed batch %d/%d.",
                batch_number + 1,
                total_batches
            )

            continue

        start_index = (
            batch_number * BATCH_SIZE
        )

        end_index = min(
            start_index + BATCH_SIZE,
            total_chunks
        )

        batch_texts = texts[
            start_index:end_index
        ]

        logger.info(
            "Embedding batch %d/%d "
            "(chunks %d-%d).",
            batch_number + 1,
            total_batches,
            start_index,
            end_index - 1
        )

        start_time = time.time()

        embeddings = model.encode(
            batch_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embeddings = embeddings.astype(
            np.float32
        )

        batch_file = (
            batch_dir
            / f"batch_{batch_number:04d}.npy"
        )

        np.save(
            batch_file,
            embeddings
        )

        completed_batches.add(
            batch_number
        )

        save_checkpoint(
            checkpoint_file,
            sorted(completed_batches),
            total_batches,
            "running"
        )

        elapsed = time.time() - start_time

        logger.info(
            "Saved batch %d/%d. "
            "Shape=%s. Time=%.2f sec.",
            batch_number + 1,
            total_batches,
            embeddings.shape,
            elapsed
        )

    # Final consolidation
    embeddings = consolidate_batches(
        batch_dir,
        total_batches,
        final_file
    )

    save_checkpoint(
        checkpoint_file,
        sorted(completed_batches),
        total_batches,
        "complete"
    )

    logger.info(
        "Completed chunk size %d. "
        "Final shape=%s.",
        chunk_size,
        embeddings.shape
    )

    print(
        f"Chunk size {chunk_size}: "
        f"complete -> {embeddings.shape}"
    )


def main():

    print()
    print("=" * 60)
    print("TinyRAG CORPUS EMBEDDING")
    print("=" * 60)
    print(
        f"Model: {MODEL_NAME}"
    )
    print(
        f"Batch size: {BATCH_SIZE}"
    )
    print(
        "Device: CPU"
    )
    print("=" * 60)

    logger.info(
        "Loading SentenceTransformer model."
    )

    # Import only after config has loaded the cache variables.
    from sentence_transformers import (
        SentenceTransformer
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(
            HF_CACHE_DIR / "sentence_transformers"
        )
    )

    logger.info(
        "Model loaded successfully."
    )

    for chunk_size in CHUNK_CONFIGS:

        embed_configuration(
            model,
            chunk_size
        )

    print()
    print("=" * 60)
    print("ALL CORPUS EMBEDDINGS COMPLETE")
    print("=" * 60)

    for chunk_size in CHUNK_CONFIGS:

        final_file = (
            EMBEDDINGS_DIR
            / f"embeddings_{chunk_size}.npy"
        )

        embeddings = np.load(
            final_file,
            mmap_mode="r"
        )

        print(
            f"{chunk_size}: "
            f"{embeddings.shape}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()