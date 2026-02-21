# Stage 1: Retrieval

## Table of Contents

- [Supported Datasets](#supported-datasets)
- [1-1 Embedding Pre-Computation](#1-1-entity-and-relation-embedding-pre-computation)
    * [Installation](#installation)
    * [Inference (Embedding Computation)](#inference-embedding-computation)
- [1-2 Retriever Development](#1-2-retriever-development)
    * [Installation](#installation-1)
    * [Training](#training)
    * [Inference](#inference)
    * [Evaluation](#evaluation)
- [1-3 Stage2 Node Re-Ranking](#1-3-stage2-node-re-ranking)
    * [Training](#training-1)
    * [Inference](#inference-1)

## Supported Datasets

We support two built-in multi-hop knowledge graph question answering (KGQA) datasets:

- `webqsp`
- `cwq`

## 1-1: Entity and Relation Embedding Pre-Computation

We first pre-compute and cache entity and relation embeddings for all samples to save time for later training and inference of retrievers.

### Installation

We use `gte-large-en-v1.5` for text encoder, hence the environment name.

```bash
conda create -n gte_large_en_v1-5 python=3.10 -y
conda activate gte_large_en_v1-5
pip install -r requirements/gte_large_en_v1-5.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

### Inference (Embedding Computation)

```bash
python emb.py -d D
```
where `D` should be a dataset mentioned in ["Supported Datasets"](#supported-datasets).

## 1-2: Retriever Development

We now train a retriever, employ it for retrieval (inference), and evaluate the retrieval results.

### Installation

```bash
conda create -n retriever python=3.10 -y
conda activate retriever
pip install -r requirements/retriever.txt
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.3
pip install pyg_lib==0.3.1 torch_scatter==2.1.2 torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### Training

```bash
python train.py -d D
```
where `D` should be a dataset mentioned in ["Supported Datasets"](#supported-datasets).

For logged learning curves, go to the corresponding Wandb interface. 

Once trained, there will be a folder in the current directory of the form `{dataset}_{time}` (e.g., `webqsp_Nov08-01:14:47/`) that stores the trained model checkpoint `cpt.pth`.

### Inference

```bash
python inference.py -p P
```
where `P` is the path to a saved model checkpoint. The predicted retrieval result will be stored in the same folder as the model checkpoint. For example, if `P` is `webqsp_Nov08-01:14:47/cpt.pth`, then the retrieval result will be saved as `webqsp_Nov08-01:14:47/retrieval_result.pth`.

### Evaluation

```bash
python eval.py -d D -p P
```
where `D` should be a dataset mentioned in ["Supported Datasets"](#supported-datasets) and `P` is the path to [inference result](#inference), e.g., `webqsp_Nov08-01:14:47/retrieval_result.pth`.

## 1-3 Stage2 Node Re-Ranking

This is the MVP implementation of the second-stage reranker:

- Build a Top-`K_t` subgraph from Stage1 triple scores.
- Reuse node features `semantic + topic_pe + DDE`.
- Train a GNN node classifier (`node BCE`) for answer entity reranking.
- Re-rank Stage1 triples with Stage2 node scores.

### Training

```bash
python train_stage2.py -p P -d D --top_k_t 500 --hidden_dim 256 --num_layers 2 --node_top_m 50
```

where:

- `P` is the Stage1 checkpoint path (e.g., `webqsp_Nov08-01:14:47/cpt.pth`)
- `D` is one of the supported datasets (`webqsp` or `cwq`)

By default, the Stage2 checkpoint is saved beside `P` as `stage2_cpt.pth`.

### Inference

```bash
python inference_stage2.py -p P --stage2_path S --max_K 500 --node_top_m 50 --alpha 0.9
```

where:

- `P` is the Stage1 checkpoint path
- `S` is the Stage2 checkpoint path (e.g., `webqsp_Nov08-01:14:47/stage2_cpt.pth`)
- local rerank is enabled by default to protect the front of Stage1 ranking:
  - keep Stage1 top-50 unchanged
  - keep Stage1 ranks 51-70 unchanged
  - replace Stage1 ranks 71-100 with top-30 candidates from Stage1 ranks 101-500, ranked by Stage2-enhanced score

Useful local-rerank args:

- `--local_rerank/--no-local_rerank` (default: enabled)
- `--lock_top_n 50`
- `--preserve_mid_start 51 --preserve_mid_end 70`
- `--replace_start 71 --replace_end 100`
- `--candidate_pool_start 101 --candidate_x 30`

By default, results are saved beside `P` as `retrieval_result_stage2.pth`.
The output keeps the original `scored_triples` field for compatibility with `reason/main.py`, and also adds Stage2-specific fields (`stage1_scored_triples`, `stage2_node_scores`, `stage2_top_nodes`, `stage2_meta`).
