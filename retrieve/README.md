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
- [1-3 Surrogate Hyperparameter Sweep (No Retraining)](#1-3-surrogate-hyperparameter-sweep-no-retraining)

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

## 1-3 Surrogate Hyperparameter Sweep (No Retraining)

This analysis script sweeps surrogate-target hyperparameters on a fixed `retrieval_result.pth` (no retraining), using:

- `delta` (near-shortest slack)
- `p_near` (cap on near-shortest triples)
- `b_type` (per-entity cap for auxiliary type-like triples)

The script uses proxy metrics on Top-`k_eval` triples:

- `Answer Recall@k_eval`
- `Path Coverage@k_eval`
- `score = 0.5 * AER + 0.5 * PathCoverage`

Evaluation mode:

- all-sample zero-fill: every question is included in the denominator; samples that cannot be evaluated for a metric are counted as `0`.

### Run

WebQSP:

```bash
python sweep_surrogate_hparams.py \
  -d webqsp \
  -p webqsp_Feb13-13:49:52/retrieval_result.pth \
  --k_eval 20 \
  --delta_list 0,1,2,3 \
  --pnear_list 50,100,200,300,500 \
  --btype_list 0,1,2,3,5 \
  --alpha 0.5 \
  --beta 0.2 \
  --out_dir webqsp_surrogate_sweep
```

CWQ:

```bash
python sweep_surrogate_hparams.py \
  -d cwq \
  -p cwq_Feb13-13:49:52/retrieval_result.pth \
  --k_eval 20 \
  --delta_list 0,1,2,3 \
  --pnear_list 50,100,200,300,500 \
  --btype_list 0,1,2,3,5 \
  --alpha 0.5 \
  --beta 0.2 \
  --out_dir cwq_surrogate_sweep
```

### Outputs

The `--out_dir` folder contains:

- `all_results.csv`: all parameter combinations.
- `top_configs.csv`: top configs sorted by `score`.
- `recommended_range.json`: recommended interval from top-ratio configs.
- `run_meta.json`: run settings and sample statistics.
