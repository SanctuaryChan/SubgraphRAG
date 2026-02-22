# 当前方法相对论文原始实现（dev@2ccd5d2）的完整说明

> 说明：本文档以 `dev` 分支提交 `2ccd5d21` 作为论文原始实现代理版本（以下简称“原始方法”），以当前 `dev`（`HEAD=01b0062`）作为“当前方法”。

---

## 1. 原始方法（2ccd5d2）是什么

原始方法的主流程是：

1. **Retriever Stage1**（单阶段三元组打分）
   - 输入：问题向量 + 实体向量 + 关系向量 + 图结构（含 topic PE 与 DDE）
   - 输出：每条候选三元组的 logit/score
   - 训练目标：**仅 BCE**（binary cross entropy with logits）
2. **Reasoning**
   - 用 `scored_100`（Top100）三元组构建提示词
   - LLM 直接生成 `ans:` 列表
   - 不做额外答案重排

原始关键代码位置：

- `retrieve/train.py`（只有 BCE 训练）
- `retrieve/src/model/retriever.py`（DDE 不区分关系，消息传播不带门控）
- `reason/main.py`（直接保存 LLM 原始 `res[0]`）

---

## 2. 当前方法总览（HEAD）

当前代码在原始方法上形成了“三层增强”：

1. **Stage1 主增强（核心）**：`BCE + pairwise hard-negative ranking` + `relation-gated DDE`
2. **Stage2 可选增强（检索后处理分支）**：节点重排 + 局部替换注入（local rerank）
3. **Reason 可选增强（答案后处理分支）**：基于证据的 `ans:` 重排（当前默认开启）

其中，与你当前实验“loss + DDE 小改 / G0-G4”直接对应的是第 1 层（Stage1 主增强）。

---

## 3. Stage1（G 系列）相对原始方法的核心变化

### 3.1 训练目标：从纯 BCE 改为 BCE + Pairwise Ranking

**原始方法**：
\[
\mathcal{L}_{\text{orig}} = \mathcal{L}_{\text{BCE}}
\]

**当前方法**：
\[
\mathcal{L}_{\text{new}} = \mathcal{L}_{\text{BCE}} + \lambda_{pw}\,\mathcal{L}_{pw}
\]

其中 `\mathcal{L}_{pw}` 基于 hardest positives 与 hard negatives 计算 margin-ranking（或 logsigmoid）：

- 正样本：`target_triple_probs > 0`
- 负样本：`target_triple_probs == 0`
- 只采样最难负例（`hard_neg_k`）
- 可只保留最难正例（`pos_cap`）
- 支持 `hinge` / `logsigmoid` 两种 pairwise 形式

实现位置：

- `retrieve/train.py`：
  - `compute_pairwise_loss(...)`
  - `loss = bce_loss + pairwise_weight * pairwise_loss`

新增可调参数（CLI）：

- `--pairwise_weight`（默认 0.2）
- `--pairwise_margin`（默认 0.1）
- `--hard_neg_k`（默认 64）
- `--pos_cap`（默认 32）
- `--pairwise_mode`（`hinge|logsigmoid`）

> 直观意义：原始 BCE 更偏“分类正确”；pairwise 明确优化“正例排在负例前面”，尤其针对 TopK 排序质量。

---

### 3.2 DDE 结构：从“关系无差别传播”改为“关系门控传播”

**原始 DDE**：
- 每条边传播权重相同，`message = x_j`

**当前 DDE**：
- 从关系 embedding 预测边门控：
  \[
  g_e = \sigma(\mathrm{MLP}(r_e))
  \]
- 消息改为：
  \[
  \mathrm{message}_e = g_e \cdot x_j
  \]

实现位置：

- `retrieve/src/model/retriever.py`
  - `PEConv.forward(..., edge_gate=None)`
  - `DDE(..., relation_gated, gate_hidden_dim, gate_dropout, relation_emb_dim)`
  - `self.relation_gate` + `edge_gate`

配置层新增字段：

- `retrieve/src/config/retriever.py`
  - `relation_gated: bool = False`
  - `gate_hidden_dim: int = 64`
  - `gate_dropout: float = 0.0`

数据集默认配置改动：

- `retrieve/configs/retriever/webqsp.yaml`
- `retrieve/configs/retriever/cwq.yaml`

均将 `relation_gated` 默认设为 `true`（并给出 gate hidden/dropout）。

> 直观意义：不是所有关系都同样可靠，关系门控让 DDE 在传播时对关键关系赋更高权重。

---

### 3.3 工程与训练日志增强（辅助但重要）

- `eval_epoch` 去掉不必要 CPU 迁移并补充空答案实体保护，减少潜在评估异常
- 训练日志新增：`loss_bce`, `loss_pairwise`, `num_pairs`, `score_gap` 等可诊断项
- 训练 CLI 新增 `--relation_gated/--no-relation_gated` 可一键回退

---

## 4. G0-G4 在当前框架中的定位（写作建议）

虽然你本次强调“不要相对 G0”，但论文写作里仍建议把 G 系列描述为同一方法族的内部消融。需要注意：

- G0-G4 的**精确参数组合**应以你当时训练命令/实验脚本为准。
- 从你现有实验结论看，G3/G4 属于“pairwise + relation-gated DDE”方向中的更强配置。

你当前记录显示（WebQSP，retrieval 视角）G4 在 Top100 的路径/证据指标更强：

- `shortest_path_triple_recall@100`：`0.873 -> 0.896`
- `gpt_triple_recall@100`：`0.842 -> 0.869`

这正对应你实际 `scored_100` 推理设置。

---

## 5. Stage2 新分支（相对原始方法新增，非 G4 必需）

原始方法中不存在 Stage2。当前代码新增了完整 Stage2 模块：

1. `retrieve/train_stage2.py`
   - 先用 Stage1 Top-`K_t` 构图
   - 复用 Stage1 节点特征（semantic + topic_pe + DDE）
   - 训练 `Stage2NodeReranker`（GraphSAGE + node BCE）
2. `retrieve/inference_stage2.py`
   - 计算节点分数
   - 支持两类边注入打分策略：
     - `fused`：传统融合
     - `structure`：结构约束注入（near-topic + bridge bonus）
   - 默认启用局部替换（local rerank），保护高置信前排

核心注入分数（structure）：
\[
score_{inject}(e)=\lambda_1 s_1(e)+\lambda_2\max(s_2(u),s_2(v))+\lambda_3\,near\_topic(e)+\lambda_4\,bridge(e)
\]

再做局部替换窗口（默认）：

- 锁定 `Top1-50`
- 替换 `86-100`
- 候补池来自 `101+`

> 这条分支是“在不重训 Stage1 情况下继续优化 Top100”的策略，但属于附加模块，不是论文原始主干的一部分。

---

## 6. Reason 阶段相对原始方法的变化

### 6.1 vLLM 初始化参数增强（工程稳定性）

`reason/llm_utils.py` 新增/调整：

- `max_model_len=max_seq_len_to_capture`
- `gpu_memory_utilization=0.95`
- `max_num_seqs=1`

作用：更稳定地处理长上下文与显存占用。

### 6.2 答案后处理重排（当前代码默认开启）

`reason/main.py` 新增证据重排逻辑：

- 解析 LLM 输出中的多行 `ans:`
- 用 TopK 三元组对每个答案做证据打分：
  \[
  score(a)=w_{count}\cdot \#hits(a)+w_{rank}\cdot\sum_i \exp\!\left(-\frac{rank_i-1}{\tau}\right)
  \]
- 按分数重排并截断（默认 top5）

新增参数：

- `--answer_rerank`（默认 True）
- `--answer_rerank_topk`（默认 100）
- `--answer_cap`（默认 5）
- `--ans_match_mode`（默认 substring）
- `--rerank_w_count`, `--rerank_w_rank`, `--rerank_rank_tau`

> 注意：这部分是“当前代码新增功能”，但你最近实验中该策略在 WebQSP 上未带来正向收益（F1/Hit@1 下降），因此若写论文主结果，建议将其定位为“探索性后处理”而非主方法贡献。

---

## 7. 从论文原始方法到当前方法的“方法学变化”总结

### 7.1 算法层面（最关键）

1. **目标函数变化**：`BCE -> BCE + Pairwise ranking`
2. **结构编码变化**：`关系无差别 DDE -> 关系门控 DDE`
3. **两阶段可扩展**：新增 Stage2 节点重排与结构注入（可选）
4. **答案侧后处理**：新增 evidence-based answer rerank（可选）

### 7.2 系统层面

1. 增加可控超参接口（pairwise 与 gate）
2. 增强训练诊断日志（便于分析 hard-negative 学习是否有效）
3. 推理命名包含 rerank tag，避免覆盖旧实验

---

## 8. 论文写作建议（如何讲“当前方法”）

如果你以 **G4 有效** 作为主线，建议把“当前方法”叙事分成：

1. **主方法（必须写）**：
   - Stage1 排序目标增强（pairwise）
   - Stage1 结构编码增强（relation-gated DDE）

2. **扩展方法（可选写）**：
   - Stage2 局部替换注入（检索后处理）
   - Answer rerank（生成后处理，作为探索项）

推荐在论文中把主贡献聚焦到“前排证据质量（Top100）提升”，因为这与你 `scored_100` 的真实推理路径直接一致。

---

## 9. 复现实用说明（相对 2ccd5d2）

### 9.1 近似回退到原始方法

```bash
python retrieve/train.py -d webqsp --pairwise_weight 0 --no-relation_gated
```

### 9.2 启用当前 Stage1 增强（G 系列母体）

```bash
python retrieve/train.py -d webqsp \
  --pairwise_weight 0.2 --pairwise_margin 0.1 \
  --hard_neg_k 64 --pos_cap 32 --pairwise_mode hinge \
  --relation_gated --gate_hidden_dim 64 --gate_dropout 0.0
```

### 9.3 Stage2 分支（可选）

```bash
python retrieve/train_stage2.py -p <stage1_cpt> -d webqsp --top_k_t 500 --node_top_m 50
python retrieve/inference_stage2.py -p <stage1_cpt> --stage2_path <stage2_cpt> -d webqsp
```

### 9.4 Reason 侧答案重排（当前默认开启，可关闭）

```bash
python reason/main.py -d webqsp --prompt_mode scored_100 -p <retrieval_result.pth>
# 关闭答案重排：
python reason/main.py -d webqsp --prompt_mode scored_100 -p <retrieval_result.pth> --no-answer_rerank
```

---

## 10. 文件级变更索引（相对 2ccd5d2，方法相关）

- Stage1 训练：`retrieve/train.py`
- Retriever/DDE：`retrieve/src/model/retriever.py`
- Retriever 配置：
  - `retrieve/src/config/retriever.py`
  - `retrieve/configs/retriever/webqsp.yaml`
  - `retrieve/configs/retriever/cwq.yaml`
- Stage2 新增：
  - `retrieve/src/model/stage2_reranker.py`
  - `retrieve/train_stage2.py`
  - `retrieve/inference_stage2.py`
- Reason 变化：
  - `reason/llm_utils.py`
  - `reason/main.py`

---

## 11. 一句话版本（可放论文贡献段）

相对论文原始实现（2ccd5d2），当前方法将检索器从“BCE 分类器 + 关系无差别 DDE”升级为“BCE+hard-negative 排序联合优化 + relation-gated DDE”，并提供 Stage2 结构注入与答案后处理扩展，从而把优化重心从全局召回转向 LLM 实际使用的 Top100 证据质量。