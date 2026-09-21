# TinyRAG Experiment Summary

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

- TinyRAG Exact Match = 37.20%
- TinyRAG Token F1 = 0.5179
- No-RAG Exact Match = 5.40%
- No-RAG Token F1 = 0.1205
- Exact Match difference = +31.80 percentage points
- Token F1 difference = +0.3974
- TinyRAG mean generation time = 10.98 s/question
- No-RAG mean generation time = 7.55 s/question

## Paired statistical analysis

- Questions paired: 500
- TinyRAG correct / baseline wrong: 163
- Baseline correct / TinyRAG wrong: 4
- Exact McNemar test: p < 0.0001
- Exact Match 95% paired-bootstrap CI for the difference: [27.60, 36.00] percentage points
- Token F1 95% paired-bootstrap CI for the difference: [0.3595, 0.4362]
- Bootstrap resamples: 10,000
- Bootstrap seed: 42

## Hardware and execution

- CPU-only experiment
- Laptop RAM: approximately 15.79 GB total
- The completed end-to-end RAG run recorded zero generation errors.
- The completed no-RAG baseline recorded zero generation errors.

## Important scope limitation

The retrieval experiment covered all three chunk sizes, but the full 500-question generation experiment was conducted using the 1024-character + Top-1 configuration. Therefore, the end-to-end results do not establish an end-to-end comparison of all three chunk sizes.
