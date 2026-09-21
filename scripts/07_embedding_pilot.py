from config import DATA_DIR, DEVICE, HF_CACHE_DIR
from logger import get_logger

import time

import pandas as pd
import psutil
from sentence_transformers import SentenceTransformer


logger = get_logger("EmbeddingPilot")


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INPUT_FILE = DATA_DIR / "chunks_768.csv"

PILOT_SIZE = 10


def main():

    print("=" * 60)
    print("TinyRAG Embedding CPU Pilot")
    print("=" * 60)

    logger.info("Starting embedding pilot.")

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    texts = (
        df["text"]
        .head(PILOT_SIZE)
        .astype(str)
        .tolist()
    )

    logger.info(
        "Loaded %d chunks for pilot.",
        len(texts)
    )

    # --------------------------------------------------------
    # Memory before model
    # --------------------------------------------------------

    ram_before = (
        psutil.virtual_memory().available
        / (1024 ** 3)
    )

    print(f"Available RAM before model: {ram_before:.2f} GB")

    # --------------------------------------------------------
    # Load embedding model on CPU
    # --------------------------------------------------------

    print("\nLoading model:")
    print(MODEL_NAME)

    start_load = time.perf_counter()

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    load_time = time.perf_counter() - start_load

    print(
        f"Model load time: {load_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Memory after model
    # --------------------------------------------------------

    ram_after = (
        psutil.virtual_memory().available
        / (1024 ** 3)
    )

    print(
        f"Available RAM after model: "
        f"{ram_after:.2f} GB"
    )

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    print("\nGenerating embeddings...")

    start_embed = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=4,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    embed_time = time.perf_counter() - start_embed

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PILOT RESULT")
    print("=" * 60)

    print("Embedding shape:", embeddings.shape)

    print(
        f"Embedding time: {embed_time:.2f} seconds"
    )

    print(
        f"Average time/chunk: "
        f"{embed_time / len(texts):.4f} seconds"
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    ram_final = (
        psutil.virtual_memory().available
        / (1024 ** 3)
    )

    print(
        f"Available RAM after embedding: "
        f"{ram_final:.2f} GB"
    )

    print(
        f"\nHF cache: {HF_CACHE_DIR}"
    )

    print("=" * 60)
    print("EMBEDDING PILOT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()