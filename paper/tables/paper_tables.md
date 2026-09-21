# TinyRAG Paper-Ready Tables


## Table 1. Retrieval performance across chunk sizes


|   Chunk size |   Corpus chunks |   Recall@1 (%) |   Recall@5 (%) |   Recall@10 (%) |    MRR |   Mean retrieval (ms) |   P95 retrieval (ms) |
|-------------:|----------------:|---------------:|---------------:|----------------:|-------:|----------------------:|---------------------:|
|          512 |            1005 |           72.8 |           93.6 |            95.2 | 0.8166 |                 0.206 |                0.269 |
|          768 |             651 |           76.4 |           94   |            97   | 0.8448 |                 0.158 |                0.209 |
|         1024 |             529 |           80.2 |           95.2 |            97.2 | 0.868  |                 0.151 |                0.197 |


## Table 2. End-to-end QA performance


| System          | Context         |   Exact Match (%) |   Token F1 |   Mean generation (s) |   Median generation (s) |   Mean available RAM (GB) |
|:----------------|:----------------|------------------:|-----------:|----------------------:|------------------------:|--------------------------:|
| No-RAG baseline | None            |               5.4 |     0.1205 |                  7.55 |                    6.45 |                      2.49 |
| TinyRAG         | 1024-char Top-1 |              37.2 |     0.5179 |                 10.98 |                   10.37 |                      2.56 |


## Table 3. Paired comparison of TinyRAG and no-RAG


| Metric      |   TinyRAG |   No-RAG |   Difference |   95% CI low |   95% CI high | Paired test      | p-value           |
|:------------|----------:|---------:|-------------:|-------------:|--------------:|:-----------------|:------------------|
| Exact Match |    0.372  |   0.054  |       0.318  |       0.276  |        0.36   | Exact McNemar    | <0.0001           |
| Token F1    |    0.5179 |   0.1205 |       0.3974 |       0.3595 |        0.4362 | Paired bootstrap | 95% CI excludes 0 |


## Statistical note
The same 500 questions were evaluated in both systems. The exact McNemar test used 163 questions where TinyRAG was correct and the baseline was incorrect, versus 4 questions showing the reverse outcome; the exact two-sided p-value was below 0.0001. Paired bootstrap confidence intervals used 10,000 resamples with seed 42.
