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
pip install vllm openai wandb ollama pyyaml transformers
# Optional fallback tokenizer loader for ModelScope snapshots:
# pip install modelscope
```

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

To run with a local Qwen3 model via vLLM, point `--local_model_path` to the downloaded snapshot and keep `--model_name` as the portable fallback model id:

```bash
python main.py \
  -d cwq \
  --prompt_mode scored_100 \
  --llm_mode sys_icl_dc \
  --llm_backend local_vllm \
  -m Qwen/Qwen3-0.6B \
  --local_model_path /data/models/Qwen3-0.6B \
  --model_alias qwen3_0_6b \
  --disable_wandb
```

KGQA benchmarking defaults to non-thinking output for Qwen3 so the generated answers remain easy to evaluate. To explicitly compare thinking mode, add `--enable_thinking`.

### Using Alternative Retrieval Results

To use alternative retrieval results,

```
python main.py -d webqsp --prompt_mode scored_100 -p P
```
where `P` is the path to the retrieval results obtained from retrieval inference, e.g., `../retrieve/webqsp_Nov08-01:14:47/retrieval_result.pth`.

### Config

Our used config for each dataset can be found in `./config`.

### Multi-LLM Benchmark

To benchmark multiple models in one run, edit `configs/model_zoo_local.yaml` first.

The default local model zoo now expects entries like:

```yaml
models:
  - alias: qwen3_0_6b
    model_name: Qwen/Qwen3-0.6B
    local_model_path: /data/models/Qwen3-0.6B
    enable_thinking: false
```

`local_model_path` is optional. If it exists, the benchmark uses it first; otherwise it falls back to `model_name`.

Then run:

```bash
python run_multi_llm_benchmark.py \
  -d cwq \
  --model_zoo configs/model_zoo_local.yaml \
  --prompt_mode scored_100 \
  --llm_mode sys_icl_dc \
  --split test \
  --max_tokens 1024 \
  --temperature 0 \
  --frequency_penalty 0.16 \
  --thres 0.0 \
  --disable_wandb
```

Run selected models only:

```bash
python run_multi_llm_benchmark.py \
  -d cwq \
  --model_zoo configs/model_zoo_local.yaml \
  --models qwen3_0_6b
```

The script writes a merged table to:

`results/KGQA/<dataset>/SubgraphRAG/leaderboard.csv`
