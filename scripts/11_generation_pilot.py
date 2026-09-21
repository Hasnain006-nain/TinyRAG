import gc
import time
from pathlib import Path

import pandas as pd
import psutil
import torch

from config import DATA_DIR, HF_CACHE_DIR
from logger import get_logger


logger = get_logger("GenerationPilot")


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

CHUNK_SIZE = 1024
TOP_K = 1

NUM_PILOT_QUESTIONS = 10

MAX_NEW_TOKENS = 32          # ← CHANGED from 64 to 32

CPU_THREADS = 4

RESULTS_DIR = (
    DATA_DIR.parent
    / "results"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "generation_pilot_10.csv"
)


# ============================================================
# MEMORY
# ============================================================

def get_memory():

    memory = psutil.virtual_memory()

    return {
        "total_gb": memory.total / (1024 ** 3),
        "available_gb": memory.available / (1024 ** 3),
        "used_percent": memory.percent,
    }


def print_memory(label):

    memory = get_memory()

    print(
        f"{label}: "
        f"available={memory['available_gb']:.2f} GB, "
        f"used={memory['used_percent']:.1f}%"
    )


# ============================================================
# PROMPT
# ============================================================

def build_prompt(question, context):

    return f"""You are answering a reading-comprehension question.

Use only the provided context.

Give the shortest correct answer possible.
Do not explain your reasoning.
Do not add information that is not supported by the context.

Context:
{context}

Question:
{question}

Answer:"""


# ============================================================
# MODEL
# ============================================================

def load_model():

    logger.info(
        "Loading model: %s",
        MODEL_NAME
    )

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    model_cache = (
        HF_CACHE_DIR
        / "generation_model"
    )

    model_cache.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=str(model_cache)
    )

    logger.info(
        "Tokenizer loaded."
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=str(model_cache),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    model.to("cpu")

    model.eval()

    logger.info(
        "Model loaded successfully."
    )

    return tokenizer, model


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    tokenizer,
    model,
    question,
    context
):

    prompt = build_prompt(
        question,
        context
    )

    # --------------------------------------------------------
    # Build the final prompt using the model's chat template.
    # We first create text, then tokenize it normally.
    # This avoids treating a BatchEncoding object as a Tensor.
    # --------------------------------------------------------

    if getattr(
        tokenizer,
        "chat_template",
        None
    ):

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        formatted_prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        encoded = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False
        )

    else:

        encoded = tokenizer(
            prompt,
            return_tensors="pt"
        )

    # Move tensors to CPU
    model_inputs = {
        key: value.to("cpu")
        for key, value in encoded.items()
    }

    input_length = (
        model_inputs["input_ids"]
        .shape[1]
    )

    start_time = time.perf_counter()

    with torch.inference_mode():

        generated = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
            eos_token_id=(
                tokenizer.eos_token_id
            )
        )

    generation_time = (
        time.perf_counter()
        - start_time
    )

    generated_tokens = generated[
        0,
        input_length:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return (
        answer,
        generation_time
    )


# ============================================================
# MAIN
# ============================================================

def main():

    torch.set_num_threads(
        CPU_THREADS
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("TinyRAG GENERATION PILOT")
    print("=" * 60)
    print(
        f"Model: {MODEL_NAME}"
    )
    print(
        f"Chunk size: {CHUNK_SIZE}"
    )
    print(
        f"Top-k: {TOP_K}"
    )
    print(
        f"Questions: {NUM_PILOT_QUESTIONS}"
    )
    print(
        f"Max new tokens: {MAX_NEW_TOKENS}"
    )
    print(
        f"CPU threads: {CPU_THREADS}"
    )
    print(
        "Device: CPU"
    )
    print("=" * 60)

    print_memory(
        "Before model loading"
    )

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    questions_file = (
        DATA_DIR
        / "eval_500.csv"
    )

    questions = pd.read_csv(
        questions_file,
        dtype={"id": str}
    )

    questions = questions.iloc[
        :NUM_PILOT_QUESTIONS
    ].copy()

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    corpus_file = (
        DATA_DIR
        / f"corpus_chunks_{CHUNK_SIZE}.csv"
    )

    chunks = pd.read_csv(
        corpus_file
    )

    # --------------------------------------------------------
    # Load retrieval results
    # --------------------------------------------------------

    retrieval_file = (
        RESULTS_DIR
        / "retrieval"
        / f"retrieval_{CHUNK_SIZE}.csv"
    )

    retrieval = pd.read_csv(
        retrieval_file,
        dtype={"question_id": str}
    )

    retrieval = retrieval[
        retrieval["question_id"].isin(
            questions["id"]
        )
    ].copy()

    retrieval = retrieval.sort_values(
        "question_index"
    )

    # --------------------------------------------------------
    # Verify 10 retrieval records
    # --------------------------------------------------------

    if len(retrieval) != NUM_PILOT_QUESTIONS:

        raise ValueError(
            f"Expected {NUM_PILOT_QUESTIONS} "
            f"retrieval records, found "
            f"{len(retrieval)}."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    tokenizer, model = load_model()

    print_memory(
        "After model loading"
    )

    # --------------------------------------------------------
    # Generate answers
    # --------------------------------------------------------

    results = []

    for _, question_row in questions.iterrows():

        question_id = str(
            question_row["id"]
        )

        question = str(
            question_row["question"]
        )

        gold_answer = str(
            question_row["answer"]
        )

        retrieval_row = retrieval[
            retrieval["question_id"]
            == question_id
        ].iloc[0]

        chunk_id = int(
            retrieval_row["top1_chunk_id"]
        )

        retrieval_score = float(
            retrieval_row["top1_score"]
        )

        context_row = chunks[
            chunks["chunk_id"] == chunk_id
        ]

        if len(context_row) != 1:

            raise ValueError(
                f"Could not find exactly one "
                f"chunk for {chunk_id}."
            )

        context = str(
            context_row.iloc[0]["text"]
        )

        print()
        print("-" * 60)
        print(
            f"Question {len(results) + 1}/"
            f"{NUM_PILOT_QUESTIONS}"
        )
        print(
            f"ID: {question_id}"
        )
        print(
            f"Question: {question}"
        )
        print(
            f"Retrieved chunk: {chunk_id}"
        )
        print(
            f"Retrieval score: "
            f"{retrieval_score:.4f}"
        )

        generation_start = (
            time.perf_counter()
        )

        try:

            answer, generation_time = (
                generate_answer(
                    tokenizer,
                    model,
                    question,
                    context
                )
            )

            generation_error = ""

        except Exception as exc:

            answer = ""

            generation_time = (
                time.perf_counter()
                - generation_start
            )

            generation_error = (
                repr(exc)
            )

            logger.exception(
                "Generation failed for %s",
                question_id
            )

        print(
            f"Generated answer: {answer}"
        )

        print(
            f"Generation time: "
            f"{generation_time:.2f} sec"
        )

        memory = get_memory()

        result = {
            "question_id": question_id,
            "question": question,
            "gold_answer": gold_answer,
            "chunk_size": CHUNK_SIZE,
            "top_k": TOP_K,
            "retrieved_chunk_id": chunk_id,
            "retrieval_score": retrieval_score,
            "retrieved_context": context,
            "generated_answer": answer,
            "generation_time_sec": generation_time,
            "available_ram_gb": (
                memory["available_gb"]
            ),
            "ram_used_percent": (
                memory["used_percent"]
            ),
            "generation_error": (
                generation_error
            )
        }

        results.append(
            result
        )

        # Save after every question.
        pd.DataFrame(
            results
        ).to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8"
        )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del model
    del tokenizer

    gc.collect()

    print_memory(
        "After pilot cleanup"
    )

    print()
    print("=" * 60)
    print("GENERATION PILOT COMPLETE")
    print("=" * 60)

    print(
        f"Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    successful_generations = sum(
        1
        for r in results
        if not r["generation_error"]
    )

    print(
        f"Successful generations: "
        f"{successful_generations}/{len(results)}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()