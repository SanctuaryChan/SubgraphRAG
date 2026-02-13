# SubgraphRAG 论文实验复现指南（中文）

本文档基于仓库代码与论文 `SubgraphRAG.pdf` 对照整理，目标是帮助你尽可能完整地复现论文中的实验（检索、推理、消融、分组评测、可解释性示例）。

## 1. 复现范围先说明

本仓库可以直接复现的核心内容：

1. SubgraphRAG 检索器训练/推理/评测（Stage-1，`retrieve/`）。
2. SubgraphRAG 基于检索结果的 LLM 推理与评测（Stage-2，`reason/`）。
3. 论文中的 SubgraphRAG 主结果与大部分消融流程（在有对应输入文件时）。

需要额外外部代码/结果的部分：

1. 论文里多个 baseline（如 SR+NSM、RoG、G-Retriever、ToG 等）并不在本仓库实现。
2. 这些 baseline 的行可通过“外部仓库跑出检索结果后接入 `reason/main.py`”复现，但前置训练/推理不在本仓库内。

## 2. 路径与目录约定（非常重要）

1. `retrieve/` 与 `reason/` 是两个阶段，命令需在对应目录执行。
2. `reason/main.py` 与评测脚本默认使用相对路径 `./scored_triples` 和 `./results/KGQA`。  
   如果你把 HuggingFace 资源下载到别处，需要软链接回 `reason/`。

示例（在仓库根目录执行）：

```bash
ln -s /your/path/scored_triples reason/scored_triples
ln -s /your/path/results reason/results
```

## 3. 环境准备

论文与仓库实际上是三套环境：

1. `retrieve` 的 embedding 预计算环境。
2. `retrieve` 的 retriever 训练环境。
3. `reason` 的 LLM 推理环境。

### 3.1 Stage-1 embedding 环境

```bash
cd retrieve
conda create -n gte_large_en_v1-5 python=3.10 -y
conda activate gte_large_en_v1-5
pip install -r requirements/gte_large_en_v1-5.txt
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

### 3.2 Stage-1 retriever 环境

```bash
cd retrieve
conda create -n retriever python=3.10 -y
conda activate retriever
pip install -r requirements/retriever.txt
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.3
pip install pyg_lib==0.3.1 torch_scatter==2.1.2 torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### 3.3 Stage-2 reasoning 环境

```bash
cd reason
conda create -n reasoner python=3.10.14 -y
conda activate reasoner
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.5.5 openai==1.50.2 wandb
```

可选：

```bash
export WANDB_MODE=offline
export OPENAI_API_KEY=YOUR_KEY   # 使用 GPT 系列时需要
```

## 4. Stage-1：检索实验复现（`retrieve/`）

### 4.1 预计算实体/关系/问题 embedding

```bash
cd retrieve
conda activate gte_large_en_v1-5
python emb.py -d webqsp
python emb.py -d cwq
```

会产生：

1. `retrieve/data_files/<dataset>/processed/{train,val,test}.pkl`
2. `retrieve/data_files/<dataset>/emb/gte-large-en-v1.5/{train,val,test}.pth`

### 4.2 训练 retriever

```bash
cd retrieve
conda activate retriever
python train.py -d webqsp
python train.py -d cwq
```

每次训练会创建时间戳目录，如 `webqsp_Nov08-01:14:47/`，其中包含 `cpt.pth`。

### 4.3 检索推理

```bash
cd retrieve
conda activate retriever
python inference.py -p webqsp_xxx/cpt.pth --max_K 500
python inference.py -p cwq_xxx/cpt.pth --max_K 500
```

输出：`<run_dir>/retrieval_result.pth`。

### 4.4 检索评测（召回）

```bash
cd retrieve
conda activate retriever
python eval.py -d webqsp -p webqsp_xxx/retrieval_result.pth --k_list 50,100,200,400
python eval.py -d cwq -p cwq_xxx/retrieval_result.pth --k_list 50,100,200,400
```

指标对应论文检索评估：`ans_recall`、`shortest_path_triple_recall`、`gpt_triple_recall`。

### 4.5 检索耗时（Table 1 的 time）

示例：

```bash
cd retrieve
conda activate retriever
/usr/bin/time -p python inference.py -p webqsp_xxx/cpt.pth --max_K 100
/usr/bin/time -p python inference.py -p cwq_xxx/cpt.pth --max_K 100
```

### 4.6 跨数据集泛化（A→B）

`inference.py` 默认读 checkpoint 内的 `config.dataset.name`，所以做 A→B 需临时改 checkpoint 配置：

```bash
cd retrieve
conda activate retriever
python - <<'PY'
import torch
src='cwq_xxx/cpt.pth'
dst='cwq_xxx/cpt_for_webqsp.pth'
cpt=torch.load(src, map_location='cpu')
cpt['config']['dataset']['name']='webqsp'
torch.save(cpt, dst)
PY
python inference.py -p cwq_xxx/cpt_for_webqsp.pth --max_K 500
python eval.py -d webqsp -p cwq_xxx/retrieval_result.pth --k_list 100
```

同理可做 `webqsp -> cwq`。

## 5. Stage-2：LLM 推理实验复现（`reason/`）

### 5.1 快速复现（用作者提供的预处理结果）

```bash
cd reason
conda activate reasoner
huggingface-cli download siqim311/SubgraphRAG --revision main --local-dir ./
```

然后可直接推理：

```bash
python main.py -d webqsp --prompt_mode scored_100
python main.py -d cwq --prompt_mode scored_100
```

### 5.2 使用你自己训练出来的检索结果

```bash
cd reason
conda activate reasoner
python main.py -d webqsp --prompt_mode scored_100 -p ../retrieve/webqsp_xxx/retrieval_result.pth
python main.py -d cwq --prompt_mode scored_100 -p ../retrieve/cwq_xxx/retrieval_result.pth
```

### 5.3 对齐论文的推理配置建议

论文正文给出的关键设置：

1. 温度与 seed 设为 0。
2. 默认 top-100 triples，部分实验用 200/500。
3. 使用 Llama3.1-8B/70B-Instruct、gpt-3.5-turbo-1106、gpt-4o-mini-2024-07-18、gpt-4o-2024-08-06。

可直接用：

```bash
# WebQSP
python main.py -d webqsp --prompt_mode scored_100 --llm_mode sys_icl_dc_repro --frequency_penalty 0.14 -m meta-llama/Meta-Llama-3.1-8B-Instruct --temperature 0 --seed 0

# CWQ
python main.py -d cwq --prompt_mode scored_100 --llm_mode sys_icl_dc_repro --frequency_penalty 0.16 -m meta-llama/Meta-Llama-3.1-8B-Instruct --temperature 0 --seed 0
```

### 5.4 结果文件位置

1. 推理输出：`reason/results/KGQA/<dataset>/SubgraphRAG/<model>/...predictions.jsonl`
2. 自动评测文件：同目录下 `eval_result_corrected.txt`、`eval_result.txt`、`*_detailed_eval_result*.jsonl`

## 6. 对照论文表/图的复现清单

下面默认你已经完成 Stage-1 与 Stage-2。

### 6.1 检索部分

1. Table 1（检索总体 recall + time）  
   用 `retrieve/eval.py` 产 recall；用 `/usr/bin/time` 统计推理耗时。
2. Table 2（按 hop 分组的检索 recall）  
   需要基于 `retrieval_result.pth` + `max_path_length` 自写分组统计脚本（仓库未提供现成脚本）。
3. Figure 3 / Figure 6（不同 K 的检索效果曲线）  
   `inference.py --max_K 500` 后，`eval.py --k_list` 设为多个 K（如 `20,50,100,200,300,400,500`）作图。
4. Table 7（按 topic entity 数量分组）  
   需自写分组统计脚本（仓库未提供现成脚本）。

### 6.2 QA 部分

1. Table 3（WebQSP/CWQ）  
   `main.py` 分别在不同模型与 `prompt_mode`（`scored_100/200/500`）下运行。
2. Table 4（WebQSP-sub/CWQ-sub）  
   `main.py` 运行后自动给出 `subset=True` 指标（即 sub 数据集指标）。
3. Table 5（按推理 hop 分组）  
   调用评测函数并设置 `eval_hops` 为 `1/2/3`（其中 `3` 代表 `>=3`）。
4. Table 6（不同 retriever + Llama3.1-8B）  
   固定 reasoner，替换 `-p` 指向不同 retriever 的结果；随机/无检索用 `prompt_mode=rand_100`、`randNoA_100`、`noevi`。
5. Table 8（truth-grounded 细分统计）  
   使用 `evaluate_results_corrected.py` 返回的 `stats` 字典。
6. Table 9（不同 retriever + GPT4o-mini）  
   同 Table 6，但 reasoner 换为 GPT4o-mini。
7. Figure 4（不同 triple 数量对 QA 的影响）  
   改 `prompt_mode` 为 `scored_50/100/200/500` 等并作图。
8. Figure 7（不同 retriever 与 triple 数量）  
   对每种 retriever 结果分别跑 `scored_K` 系列并作图。
9. Appendix F（可解释性案例）  
   直接从 `predictions.jsonl` 抽样整理示例。

## 7. 分组评测脚本示例

### 7.1 Table 5（按 hop）

```bash
cd reason
conda activate reasoner
python - <<'PY'
from metrics.evaluate_results_corrected import eval_results
pred = "results/KGQA/webqsp/SubgraphRAG/Meta-Llama-3.1-8B-Instruct/scored_100-sys_icl_dc_repro-0.14-thres_0.0-test-predictions.jsonl"
for hop in [1, 2, 3]:
    out = eval_results(pred, cal_f1=True, subset=True, eval_hops=hop)
    print("hop", hop, "Hit@1/MacroF1/MacroP/MacroR =", out[:4])
PY
```

### 7.2 Table 8（truth-grounded 细粒度统计）

```bash
cd reason
conda activate reasoner
python - <<'PY'
from metrics.evaluate_results_corrected import eval_results
pred = "results/KGQA/cwq/SubgraphRAG/gpt-4o-mini-2024-07-18/scored_100-sys_icl_dc_repro-0.16-thres_0.0-test-predictions.jsonl"
out = eval_results(pred, cal_f1=True, subset=False, eval_hops=-1)
print("HalScore =", out[12])
print("Detailed stats =", out[13])
PY
```

## 8. baseline 结果接入格式（用于 Table 6/9）

`reason/preprocess/prepare_data.py` 支持两类检索文件格式（按样本 id 建索引）：

1. `{"scored_triples": [(h,r,t,score), ...]}`  
2. `{"triples": [(h,r,t), ...]}`（代码会自动映射到 `scored_triples`）

所以如果你在外部仓库跑了 baseline，只需把输出转换成上述结构，再通过 `main.py -p <path>` 接入。

## 9. 常见问题与坑

1. `retrieve/emb.py`、`retrieve/train.py`、`retrieve/inference.py` 默认是 `cuda:0`，无 CPU 路径。
2. `reason/main.py` 会强依赖 `./results/KGQA/...RoG.../predictions.jsonl` 作为基础 QA 文件来源，所以 `reason/` 下资源目录必须齐全。
3. `inference.py` 每次都会把结果写成 `<run_dir>/retrieval_result.pth`，注意备份避免覆盖。
4. `main.py` 默认会 `wandb.init(...)`，如不想联网上传请设 `WANDB_MODE=offline`。
5. GPT 线上模型随时间可能下线/升级。要做“论文严格同版复现”，尽量使用论文给出的具体版本名。

## 10. 推荐复现顺序（最稳）

1. 先在 `reason/` 下载作者预处理结果，验证 `main.py` 能跑通。
2. 用论文推荐设置先复现 SubgraphRAG + Llama3.1-8B 的 WebQSP/CWQ。
3. 再切换到 GPT4o-mini/GPT4o，复现主结果和 K 扩展（100/200/500）。
4. 完成 Stage-1 自训练 retriever，替换 `-p` 走完整闭环。
5. 最后做分组评测（hop/topic count）和 retriever ablation（Table 6/9、Figure 7）。
