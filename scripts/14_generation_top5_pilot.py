import gc
import json
import time

import pandas as pd
import psutil
import torch

from config import DATA_DIR, HF_CACHE_DIR
from logger import get_logger


logger = get_logger("GenerationTop5Pilot")


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

CHUNK_SIZE = 1024
TOP_K = 5

NUM_PILOT_QUESTIONS = 10

MAX_NEW_TOKENS = 32

CPU_THREADS = 4

RESULTS_DIR = (
    DATA_DIR.parent / "results"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "generation_top5_pilot_10.csv"
)


# ============================================================
# MEMORY
# ============================================================

def get_memory():

    memory = psutil.virtual_memory()

    return {
        "available_gb":
            memory.available / (1024 ** 3),
        "used_percent":
            memory.percent
    }


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    question,
    contexts
):

    joined_context = "\n\n".join(
        [
            f"Context {i + 1}:\n{context}"
            for i, context in enumerate(contexts)
        ]
    )

    return f"""Answer the question using only the provided contexts.

Return ONLY the shortest answer phrase that directly answers the question.

Do not explain your answer.
Do not repeat the question.
Do not write a full sentence unless a full sentence is necessary.
For a person question, give the person's name.
For a location question, give the location.
For a year/date question, give the year/date.
For a "what" question, give the requested fact or phrase.

Contexts:

{joined_context}

Question:
{question}

Answer:"""


# ============================================================
# MODEL
# ============================================================

def load_model():

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer
    )

    model_cache = (
        HF_CACHE_DIR / "generation_model"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=str(model_cache)
        )
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=str(model_cache),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
    )

    model.to("cpu")
    model.eval()

    return tokenizer, model


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    tokenizer,
    model,
    question,
    contexts
):

    prompt = build_prompt(
        question,
        contexts
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

    start = time.perf_counter()

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
        time.perf_counter() - start
    )

    generated_tokens = generated[
        0,
        input_length:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return answer, generation_time


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
    print("TinyRAG TOP-5 GENERATION PILOT")
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
        "Device: CPU"
    )
    print("=" * 60)

    questions = pd.read_csv(
        DATA_DIR / "eval_500.csv",
        dtype={"id": str}
    ).iloc[
        :NUM_PILOT_QUESTIONS
    ].copy()

    retrieval = pd.read_csv(
        RESULTS_DIR
        / "retrieval"
        / f"retrieval_{CHUNK_SIZE}.csv",
        dtype={"question_id": str}
    )

    retrieval = retrieval[
        retrieval["question_id"].isin(
            questions["id"]
        )
    ]

    corpus = pd.read_csv(
        DATA_DIR
        / f"corpus_chunks_{CHUNK_SIZE}.csv"
    )

    tokenizer, model = load_model()

    print(
        f"RAM after model loading: "
        f"{get_memory()['available_gb']:.2f} GB available"
    )

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

        chunk_ids = json.loads(
            retrieval_row["top10_chunk_ids"]
        )[:TOP_K]

        scores = json.loads(
            retrieval_row["top10_scores"]
        )[:TOP_K]

        contexts = []

        for chunk_id in chunk_ids:

            match = corpus[
                corpus["chunk_id"]
                == int(chunk_id)
            ]

            if len(match) != 1:

                raise ValueError(
                    f"Missing chunk {chunk_id}"
                )

            contexts.append(
                str(
                    match.iloc[0]["text"]
                )
            )

        print()
        print("-" * 60)
        print(
            f"Question {len(results) + 1}/"
            f"{NUM_PILOT_QUESTIONS}"
        )
        print(
            f"Question: {question}"
        )
        print(
            f"Top-{TOP_K} chunks: {chunk_ids}"
        )

        answer, generation_time = (
            generate_answer(
                tokenizer,
                model,
                question,
                contexts
            )
        )

        print(
            f"Generated answer: {answer}"
        )

        print(
            f"Generation time: "
            f"{generation_time:.2f} sec"
        )

        memory = get_memory()

        results.append(
            {
                "question_id": question_id,
                "question": question,
                "gold_answer": gold_answer,
                "chunk_size": CHUNK_SIZE,
                "top_k": TOP_K,
                "retrieved_chunk_ids":
                    json.dumps(chunk_ids),
                "retrieval_scores":
                    json.dumps(scores),
                "generated_answer": answer,
                "generation_time_sec":
                    generation_time,
                "available_ram_gb":
                    memory["available_gb"],
                "ram_used_percent":
                    memory["used_percent"]
            }
        )

        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8"
        )

    del model
    del tokenizer

    gc.collect()

    print()
    print("=" * 60)
    print("TOP-5 GENERATION PILOT COMPLETE")
    print("=" * 60)
    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()