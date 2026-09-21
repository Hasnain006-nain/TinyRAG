import gc
import json
import re
import string
import time

import pandas as pd
import psutil
import torch

from config import DATA_DIR, HF_CACHE_DIR
from logger import get_logger


logger = get_logger("BaselineEvaluation")


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

MAX_NEW_TOKENS = 32
CPU_THREADS = 4
NUM_QUESTIONS = 500


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = DATA_DIR.parent

RESULTS_DIR = (
    PROJECT_DIR / "results" / "baseline"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

QUESTIONS_FILE = (
    DATA_DIR / "eval_500.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "baseline_500.csv"
)

CHECKPOINT_FILE = (
    RESULTS_DIR
    / "checkpoint_baseline.json"
)


# ============================================================
# MEMORY
# ============================================================

def get_memory():

    memory = psutil.virtual_memory()

    return {
        "total_gb":
            memory.total / (1024 ** 3),
        "available_gb":
            memory.available / (1024 ** 3),
        "used_percent":
            memory.percent
    }


def print_memory(label):

    memory = get_memory()

    print(
        f"{label}: "
        f"available="
        f"{memory['available_gb']:.2f} GB, "
        f"used="
        f"{memory['used_percent']:.1f}%"
    )


# ============================================================
# BASELINE PROMPT
# ============================================================

def build_prompt(question):

    return f"""Answer the question.

Return ONLY the shortest answer phrase that directly answers the question.

Do not explain your answer.
Do not repeat the question.
Do not add unnecessary information.
For a person question, give the person's name.
For a location question, give the location.
For a year/date question, give the year/date.
For a "what" question, give the requested fact or phrase.

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
        AutoTokenizer
    )

    model_cache = (
        HF_CACHE_DIR
        / "generation_model"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=str(model_cache)
        )
    )

    logger.info(
        "Tokenizer loaded."
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=str(model_cache),
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )
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
    question
):

    prompt = build_prompt(
        question
    )

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
# CHECKPOINT
# ============================================================

def load_completed_ids():

    if not OUTPUT_FILE.exists():

        return set()

    try:

        df = pd.read_csv(
            OUTPUT_FILE,
            dtype={
                "question_id": str
            }
        )

        if "question_id" not in df.columns:

            return set()

        return set(
            df["question_id"].astype(str)
        )

    except Exception as exc:

        logger.warning(
            "Could not load existing baseline "
            "results: %s",
            exc
        )

        return set()


def save_checkpoint(
    completed_count,
    status
):

    checkpoint = {
        "status": status,
        "completed_questions":
            completed_count,
        "total_questions":
            NUM_QUESTIONS,
        "model":
            MODEL_NAME,
        "updated_at":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }

    temp_file = (
        CHECKPOINT_FILE.with_suffix(
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
        CHECKPOINT_FILE
    )


def save_result(result):

    result_df = pd.DataFrame(
        [result]
    )

    file_exists = (
        OUTPUT_FILE.exists()
    )

    result_df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8"
    )


# ============================================================
# SQUAD METRICS
# ============================================================

def normalize_answer(text):

    text = str(text).lower()

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text
    )

    return " ".join(
        text.split()
    )


def exact_match(
    prediction,
    gold
):

    return int(
        normalize_answer(prediction)
        == normalize_answer(gold)
    )


def token_f1(
    prediction,
    gold
):

    prediction_tokens = (
        normalize_answer(prediction)
        .split()
    )

    gold_tokens = (
        normalize_answer(gold)
        .split()
    )

    if (
        len(prediction_tokens) == 0
        or len(gold_tokens) == 0
    ):

        return float(
            prediction_tokens
            == gold_tokens
        )

    common_tokens = (
        set(prediction_tokens)
        & set(gold_tokens)
    )

    common_count = sum(
        min(
            prediction_tokens.count(token),
            gold_tokens.count(token)
        )
        for token in common_tokens
    )

    if common_count == 0:

        return 0.0

    precision = (
        common_count
        / len(prediction_tokens)
    )

    recall = (
        common_count
        / len(gold_tokens)
    )

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    torch.set_num_threads(
        CPU_THREADS
    )

    print()
    print("=" * 65)
    print("TinyRAG NO-RAG BASELINE")
    print("=" * 65)
    print(
        f"Model:        {MODEL_NAME}"
    )
    print(
        f"Questions:    {NUM_QUESTIONS}"
    )
    print(
        f"Max tokens:   {MAX_NEW_TOKENS}"
    )
    print(
        f"CPU threads:  {CPU_THREADS}"
    )
    print(
        "Retrieval:    NONE"
    )
    print(
        "Device:       CPU"
    )
    print("=" * 65)

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    if not QUESTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Missing questions file:\n"
            f"{QUESTIONS_FILE}"
        )

    questions = pd.read_csv(
        QUESTIONS_FILE,
        dtype={"id": str}
    )

    if len(questions) != NUM_QUESTIONS:

        raise ValueError(
            f"Expected {NUM_QUESTIONS} questions, "
            f"found {len(questions)}."
        )

    # --------------------------------------------------------
    # Resume information
    # --------------------------------------------------------

    completed_ids = (
        load_completed_ids()
    )

    print()
    print(
        f"Already completed: "
        f"{len(completed_ids)}/{NUM_QUESTIONS}"
    )

    if (
        len(completed_ids)
        == NUM_QUESTIONS
    ):

        print()
        print(
            "All baseline questions are "
            "already complete."
        )

        save_checkpoint(
            NUM_QUESTIONS,
            "complete"
        )

        return

    save_checkpoint(
        len(completed_ids),
        "running"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print_memory(
        "Before model loading"
    )

    tokenizer, model = load_model()

    print_memory(
        "After model loading"
    )

    # --------------------------------------------------------
    # Run baseline
    # --------------------------------------------------------

    run_start = time.perf_counter()

    for question_index, question_row in (
        questions.iterrows()
    ):

        question_id = str(
            question_row["id"]
        )

        if question_id in completed_ids:

            continue

        question = str(
            question_row["question"]
        )

        gold_answer = str(
            question_row["answer"]
        )

        print()
        print("-" * 65)
        print(
            f"Question "
            f"{question_index + 1}/"
            f"{NUM_QUESTIONS}"
        )
        print(
            f"ID: {question_id}"
        )
        print(
            f"Question: {question}"
        )

        generation_error = ""

        try:

            generated_answer, generation_time = (
                generate_answer(
                    tokenizer,
                    model,
                    question
                )
            )

        except Exception as exc:

            generated_answer = ""

            generation_time = 0.0

            generation_error = repr(
                exc
            )

            logger.exception(
                "Baseline generation failed "
                "for %s",
                question_id
            )

        em = exact_match(
            generated_answer,
            gold_answer
        )

        f1 = token_f1(
            generated_answer,
            gold_answer
        )

        memory = get_memory()

        result = {
            "question_id":
                question_id,
            "question_index":
                int(question_index),
            "question":
                question,
            "gold_answer":
                gold_answer,
            "generated_answer":
                generated_answer,
            "exact_match":
                em,
            "f1":
                f1,
            "model":
                MODEL_NAME,
            "retrieval":
                "none",
            "generation_time_sec":
                generation_time,
            "available_ram_gb":
                memory["available_gb"],
            "ram_used_percent":
                memory["used_percent"],
            "generation_error":
                generation_error
        }

        # Save immediately
        save_result(
            result
        )

        completed_ids.add(
            question_id
        )

        save_checkpoint(
            len(completed_ids),
            "running"
        )

        print(
            f"Generated answer: "
            f"{generated_answer}"
        )

        print(
            f"Exact Match: {em}"
        )

        print(
            f"F1: {f1:.4f}"
        )

        print(
            f"Generation time: "
            f"{generation_time:.2f} sec"
        )

        print(
            f"Available RAM: "
            f"{memory['available_gb']:.2f} GB"
        )

        # Progress every 25 questions
        if (
            len(completed_ids) % 25 == 0
            or len(completed_ids)
            == NUM_QUESTIONS
        ):

            elapsed = (
                time.perf_counter()
                - run_start
            )

            avg_time = (
                elapsed
                / max(len(completed_ids), 1)
            )

            remaining = (
                NUM_QUESTIONS
                - len(completed_ids)
            )

            eta_minutes = (
                remaining
                * avg_time
                / 60.0
            )

            logger.info(
                "Progress: %d/%d | "
                "Average %.2f sec/question | "
                "Estimated remaining %.1f min",
                len(completed_ids),
                NUM_QUESTIONS,
                avg_time,
                eta_minutes
            )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    save_checkpoint(
        NUM_QUESTIONS,
        "complete"
    )

    del model
    del tokenizer

    gc.collect()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    results = pd.read_csv(
        OUTPUT_FILE
    )

    print()
    print("=" * 65)
    print("NO-RAG BASELINE COMPLETE")
    print("=" * 65)

    print(
        f"Questions: {len(results)}"
    )

    print(
        f"Exact Match: "
        f"{results['exact_match'].mean():.4f}"
    )

    print(
        f"Token F1: "
        f"{results['f1'].mean():.4f}"
    )

    print(
        f"Mean generation time: "
        f"{results['generation_time_sec'].mean():.2f} sec"
    )

    print(
        f"Median generation time: "
        f"{results['generation_time_sec'].median():.2f} sec"
    )

    print(
        f"Mean available RAM: "
        f"{results['available_ram_gb'].mean():.2f} GB"
    )

    print()
    print(
        "Results saved to:"
    )
    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Checkpoint saved to:"
    )
    print(
        CHECKPOINT_FILE
    )

    print("=" * 65)


if __name__ == "__main__":
    main()