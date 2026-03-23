# Stage 2: Reasoning

## Table of Contents

* [Installation](#installation)
* [Pre-processed Results for Reproducibility](#pre-processed-results-for-reproducibility)
* [Inference with LLMs](#inference-with-llms)

## Installation

```bash
conda create -n reasoner python=3.10.14 -y
conda activate reasoner
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install openai==1.50.2 wandb pyyaml
```

Install `vllm` in the same environment only if you want to use `llm_backend=local_vllm`.
For Qwen3.5 multimodal checkpoints, prefer a newer standalone `vllm serve` setup and keep this benchmark environment as the client.

## Reasoning (Inference)

### Using Pre-Processed Retrieval Results for Reproducibility

We provide pre-processed results for reproducibility of the paper experiments. To download them

```bash
huggingface-cli download siqim311/SubgraphRAG --revision main --local-dir ./
```

- `scored_triples` stores the pre-processed retrieval results.
- `results/KGQA` stores the reasoning results.

After downloading the pre-processed results, one can run `main.py` with proper paramerters. For example,

```
python main.py -d webqsp --prompt_mode scored_100
python main.py -d cwq --prompt_mode scored_100
```

### Using Alternative Retrieval Results

To use alternative retrieval results,

```
python main.py -d webqsp --prompt_mode scored_100 -p P
```
where `P` is the path to the retrieval results obtained from retrieval inference, e.g., `../retrieve/webqsp_Nov08-01:14:47/retrieval_result.pth`.

### Config

Our used config for each dataset can be found in `./config`.

### Multi-LLM Benchmark (Local Models)

To benchmark multiple local models in one run, edit `configs/model_zoo_local.yaml` first.

For Qwen3.5 local weights, start a local vLLM OpenAI-compatible server manually. The client benchmark code assumes text-only usage and recommends `--language-model-only`.

Example:

```bash
vllm serve /data/models/Qwen3.5-4B \
  --served-model-name qwen35_4b \
  --tensor-parallel-size 1 \
  --language-model-only
```

`request_model_name` in `configs/model_zoo_local.yaml` must match the server's `--served-model-name`.

Then run:

```bash
python run_multi_llm_benchmark.py \
  -d cwq \
  --model_zoo configs/model_zoo_local.yaml \
  --prompt_mode scored_100 \
  --llm_mode sys_icl_dc_repro \
  --split test \
  --max_tokens 4000 \
  --temperature 0 \
  --frequency_penalty 0.16 \
  --thres 0.0 \
  --disable_wandb \
  --skip_existing
```

Run selected models only:

```bash
python run_multi_llm_benchmark.py \
  -d webqsp \
  --model_zoo configs/model_zoo_local.yaml \
  --models qwen35_4b
```

The script writes a merged table to:

`results/KGQA/<dataset>/SubgraphRAG/leaderboard.csv`
