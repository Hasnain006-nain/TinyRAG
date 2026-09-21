import pandas as pd

from config import DATA_DIR
from logger import get_logger


logger = get_logger("Corpus")


INPUT_FILE = DATA_DIR / "eval_500.csv"

DOCUMENTS_FILE = DATA_DIR / "documents.csv"
QUESTION_DOC_MAP_FILE = DATA_DIR / "question_doc_map.csv"

CHUNK_CONFIGS = [
    (512, 128),
    (768, 128),
    (1024, 128),
]


def chunk_text(text, chunk_size, overlap):

    chunks = []

    start = 0
    chunk_number = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunks.append(
            {
                "chunk_number": chunk_number,
                "start": start,
                "end": end,
                "text": text[start:end],
            }
        )

        chunk_number += 1

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def build_unique_documents(df):

    documents = (
        df[
            ["title", "context"]
        ]
        .drop_duplicates(
            subset=["context"]
        )
        .reset_index(drop=True)
    )

    documents.insert(
        0,
        "doc_id",
        range(len(documents))
    )

    return documents


def build_question_document_mapping(
    questions,
    documents
):

    context_to_doc_id = dict(
        zip(
            documents["context"],
            documents["doc_id"],
        )
    )

    mapping = questions[
        [
            "id",
            "title",
            "question",
            "answer",
            "answer_start",
            "context",
        ]
    ].copy()

    mapping["doc_id"] = mapping["context"].map(
        context_to_doc_id
    )

    if mapping["doc_id"].isna().any():
        raise ValueError(
            "Some questions could not be mapped "
            "to a document."
        )

    mapping["answer_end"] = (
        mapping["answer_start"]
        + mapping["answer"].str.len()
    )

    return mapping[
        [
            "id",
            "title",
            "question",
            "answer",
            "answer_start",
            "answer_end",
            "doc_id",
        ]
    ]


def build_chunks(documents, chunk_size, overlap):

    records = []

    global_chunk_id = 0

    for _, document in documents.iterrows():

        doc_id = int(document["doc_id"])

        chunks = chunk_text(
            str(document["context"]),
            chunk_size,
            overlap,
        )

        for chunk in chunks:

            records.append(
                {
                    "chunk_id": global_chunk_id,
                    "doc_id": doc_id,
                    "title": document["title"],
                    "chunk_number": chunk["chunk_number"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"],
                }
            )

            global_chunk_id += 1

    return pd.DataFrame(records)


def main():

    logger.info("Starting corrected RAG corpus construction.")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing evaluation file: {INPUT_FILE}"
        )

    questions = pd.read_csv(INPUT_FILE)

    logger.info(
        "Loaded %d evaluation questions.",
        len(questions),
    )

    # --------------------------------------------------------
    # Unique documents
    # --------------------------------------------------------

    documents = build_unique_documents(
        questions
    )

    documents.to_csv(
        DOCUMENTS_FILE,
        index=False,
        encoding="utf-8",
    )

    logger.info(
        "Unique documents: %d",
        len(documents),
    )

    # --------------------------------------------------------
    # Question -> document mapping
    # --------------------------------------------------------

    question_doc_map = (
        build_question_document_mapping(
            questions,
            documents,
        )
    )

    question_doc_map.to_csv(
        QUESTION_DOC_MAP_FILE,
        index=False,
        encoding="utf-8",
    )

    logger.info(
        "Question-document mapping saved."
    )

    # --------------------------------------------------------
    # Build chunk corpora
    # --------------------------------------------------------

    for chunk_size, overlap in CHUNK_CONFIGS:

        logger.info(
            "Creating corpus chunks: "
            "size=%d, overlap=%d",
            chunk_size,
            overlap,
        )

        chunks = build_chunks(
            documents,
            chunk_size,
            overlap,
        )

        output_file = (
            DATA_DIR
            / f"corpus_chunks_{chunk_size}.csv"
        )

        chunks.to_csv(
            output_file,
            index=False,
            encoding="utf-8",
        )

        logger.info(
            "Saved %d chunks to %s",
            len(chunks),
            output_file,
        )

    print()
    print("=" * 60)
    print("CORRECTED RAG CORPUS COMPLETE")
    print("=" * 60)

    print(
        f"Questions: {len(questions):,}"
    )

    print(
        f"Unique documents: {len(documents):,}"
    )

    print(
        f"Documents file: {DOCUMENTS_FILE}"
    )

    print(
        f"Question-document mapping: "
        f"{QUESTION_DOC_MAP_FILE}"
    )

    for chunk_size, _ in CHUNK_CONFIGS:

        output_file = (
            DATA_DIR
            / f"corpus_chunks_{chunk_size}.csv"
        )

        chunks = pd.read_csv(
            output_file
        )

        print(
            f"Corpus chunks {chunk_size}: "
            f"{len(chunks):,}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()