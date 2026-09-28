# TinyRAG

**CPU-only retrieval-augmented generation on a 16 GB laptop**

Available URL: https://github.com/Hasnain006-nain/TinyRAG

TinyRAG is a compact research repository for evaluating retrieval-augmented generation (RAG) under a modest local hardware budget. The project measures a dense-retrieval pipeline on 500 sampled SQuAD validation questions using a CPU-only laptop with 16 GB of RAM. The goal is not to claim a new RAG architecture or state-of-the-art question answering, but to provide a transparent baseline for retrieval quality, answer accuracy, and generation cost in a low-resource setting.

## Main Result

TinyRAG uses `sentence-transformers/all-MiniLM-L6-v2` for retrieval and `HuggingFaceTB/SmolLM2-1.7B-Instruct` for generation. The matched no-RAG baseline uses the same generator and decoding settings, but receives only the question.

| System | Context | Exact Match | Token F1 | Mean generation time | Mean available system RAM |
| --- | --- | ---: | ---: | ---: | ---: |
| No-RAG baseline | Question only | 5.4% | 0.1205 | 7.55 s/question | 2.49 GB |
| TinyRAG | 1024-character top-1 chunk | 37.2% | 0.5179 | 10.98 s/question | 2.56 GB |

The exact-match gain is **31.8 percentage points**, with a token-F1 gain of **0.3974**. Mean generation time increases by **3.43 seconds** per question.

## Retrieval Results

Retrieval is evaluated on a closed corpus constructed from 447 SQuAD source passages associated with the sampled questions. A retrieval hit means that the retrieved chunk contains the reference answer span.

| Chunk size | Corpus chunks | Recall@1 | Recall@5 | Recall@10 | MRR | Mean ranking time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 characters | 1005 | 72.8% | 93.6% | 95.2% | 0.8166 | 0.206 ms |
| 768 characters | 651 | 76.4% | 94.0% | 97.0% | 0.8448 | 0.158 ms |
| 1024 characters | 529 | 80.2% | 95.2% | 97.2% | 0.8680 | 0.151 ms |

The reported retrieval timing measures in-memory dot-product ranking over normalized embeddings. It excludes query embedding and language-model generation.

## Repository Contents

```text
TinyRAG/
├── figures/                         # Architecture and result figures
├── paper/                           # Paper-oriented summaries and tables
├── results/                         # Aggregate results and system profile
│   ├── comparison/                  # RAG vs no-RAG paired summary
│   ├── final/                       # Final combined result tables
│   ├── retrieval/                   # Retrieval metrics by chunk size
│   ├── PER_QUESTION_OUTPUTS_NOTE.txt
│   ├── observed_environment_versions.txt
│   ├── system_profile.csv
│   └── system_profile.json
├── scripts/                         # Numbered experiment pipeline scripts
├── requirements.txt
└── README.md
```

## Reproducibility Notes

This repository includes scripts, aggregate result files, paper tables, figures, and the recorded system profile used for the reported measurements. It does not include Hugging Face caches, model weights, embedding `.npy` files, checkpoints, logs, or large machine-specific artifacts.

Important details:

- The evaluation uses 500 sampled SQuAD validation questions with random seed 42.
- The corpus is closed: it is built from the sampled SQuAD source passages, not from a large open-domain collection.
- The 1024-character configuration was selected from the same evaluated sample, so the chunk-size finding should be read as descriptive for this setup.
- `available_ram_gb` reports available system RAM during measurement. It is not peak process memory and should not be interpreted as the model memory footprint.
- Exact immutable Hugging Face model revisions were not recorded in the experiment files. The model identifiers used by the scripts are documented in `results/observed_environment_versions.txt`.
- Original per-question generation CSV files are not included in this repository. The note at `results/PER_QUESTION_OUTPUTS_NOTE.txt` documents the expected file names and schema if those outputs are recovered.

## Reproduction Pipeline

Install dependencies:

```bash
git clone https://github.com/Hasnain006-nain/TinyRAG.git
cd TinyRAG
pip install -r requirements.txt
```

Prepare the dataset and chunks:

```bash
python scripts/01_environment.py
python scripts/04_prepare_dataset.py
python scripts/05_verify_dataset.py
python scripts/06_create_chunks.py
python scripts/08_create_gold_mapping.py
```

Run embedding and retrieval evaluation:

```bash
python scripts/09_embed_corpus.py
python scripts/10_retrieval_eval.py
```

Run generation and paired analysis:

```bash
python scripts/18_rag_eval.py
python scripts/19_baseline_eval.py
python scripts/20_compare_rag_baseline.py
python scripts/21_make_results_table.py
python scripts/22_make_figures.py
python scripts/23_make_paper_tables.py
python scripts/24_system_profile.py
```

## Hardware Profile

The recorded system profile reports:

- CPU: Intel64 Family 6 Model 166 Stepping 1, GenuineIntel
- Physical cores: 4
- Logical threads: 8
- System RAM: 15.792 GB
- CUDA available: false
- PyTorch device: CPU
- PyTorch threads: 4
- Python: 3.11.16
- PyTorch: 2.14.0+cpu

See `results/system_profile.json` and `results/observed_environment_versions.txt` for the recorded environment details and limitations.

## Citation

```bibtex
@misc{haider2026tinyrag,
  author       = {Haider, Hasnain},
  title        = {TinyRAG: Evaluating Retrieval-Augmented Generation on a 16 GB CPU-Only Laptop},
  year         = {2026},
  howpublished = {GitHub repository and manuscript prepared for the AAAI-27 Student Abstract and Poster Program},
  url          = {https://github.com/Hasnain006-nain/TinyRAG},
  note         = {Unpublished manuscript; update citation upon confirmed publication}
}
```

## License

This repository is released under the MIT License.
