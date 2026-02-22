  ## 第二步实施计划：Reason 侧答案证据重排（Hit@1 优先，兼顾 F1）

  ### Summary

  在不改 Retriever 的前提下，仅改 reason 推理后处理：

  - 对 LLM 生成的多条 ans: 做证据驱动重排（命中数 + rank 加权）。
  - 默认启用、默认最多保留 5 条答案。
  - 目标：优先提升 Hit@1，同时尽量维持/提升 Macro F1。

  ———

  ## 1) 变更范围与文件

  ### A. reason/main.py（主改动）

  在当前循环中（reason/main.py:148 附近）res = llm_inf_all(...) 之后、save_checkpoint(...) 之前插入后处理流程：

  1. 解析 res[0] 中的 ans: 行。
  2. 使用 each_qa["scored_triplets"] 计算每个答案的证据分数。
  3. 按分数重排答案；截断到 top-5。
  4. 将重排后的文本重新写回 each_qa["prediction"]。

  同时新增 CLI 参数（带默认值，避免误调用旧逻辑）：

  - --answer_rerank（bool，默认 True）
  - --answer_rerank_topk（int，默认 100，仅用前 100 条 triplets 打分）
  - --answer_cap（int，默认 5）
  - --ans_match_mode（str，默认 substring，可选 substring|normalized_exact）
  - --rerank_w_count（float，默认 1.0）
  - --rerank_w_rank（float，默认 1.0）
  - --rerank_rank_tau（float，默认 20.0，rank 衰减温度）

  > 注：run_name 拼接这些关键参数（简写）以便 W&B 区分实验。

  ———

  ## 2) 证据重排算法（决策已锁定）

  ### 输入

  - prediction_text：LLM 原始回答（多行，含 ans:）。
  - scored_triplets：每题 top-N 三元组，默认取前 100。

  ### 解析规则

  - 仅保留以 ans: 开头行（兼容前后空格与大小写）。
  - 去重：按 normalize(ans_text) 去重，保留首次出现项。
  - 若无 ans: 或只有 1 条，直接返回原文（不改）。

  ### 匹配与打分

  对每个候选答案 a：

  1. 在 triplets 前 K=answer_rerank_topk 中遍历，若 a 命中 (h,r,t) 的 h 或 t（默认 substring + normalize）则记为命中。
  2. 统计：
      - count_hits(a)：命中条数
      - rank_score(a) = Σ exp(-(rank_i-1)/tau)（rank 从 1 开始）
  3. 综合分：
      - score(a) = w_count * count_hits(a) + w_rank * rank_score(a)

  ### 排序稳定性

  - 主键：score 降序
  - 次键：count_hits 降序
  - 再次键：原始出现顺序升序（稳定，避免无谓抖动）

  ### 输出构造

  - 取前 answer_cap=5 条。
  - 每行格式统一为 ans: {text}。
  - 若原回答含非 ans: 说明文本，不保留（与当前评测接口一致，减少噪声）。

  ———

  ## 3) 兼容与失败模式

  ### 兼容性

  - 不修改 retrieve 输出格式；仅读 each_qa["scored_triplets"]（当前已有）。
  - 不改变 metrics 脚本；仅改变 prediction 内容顺序与长度。

  ### 失败保护

  - scored_triplets 缺失/空：退化为只做去重 + 截断（保持原顺序）。
  - 命中全为 0：保持原顺序，仅截断。
  - ans: not available：若存在且有其它有效答案，排最后；若唯一答案则保留。

  ———

  ## 4) 评测与验收标准

  ### 实验矩阵（小网格，仅 4 组）

  固定 Retriever（建议先用 G3，再复验 G4）：

  1. Baseline：--answer_rerank False（或等价关闭）
  2. Rerank-only：开启重排，answer_cap=5
  3. Rerank+更保守：answer_cap=3
  4. Rerank+不截断：answer_cap=0（表示不限制，代码需支持）

  其余参数固定：

  - --prompt_mode scored_100
  - --frequency_penalty 0.16（第一轮不改）
  - 同一模型与 seed

  ### 验收门槛（Hit@1 优先）

  - 主目标：Hit@1 相对 baseline 提升 >= 0.5pt
  - 约束：Macro F1 不低于 baseline 0.3pt 以上
  - 若 Hit@1 提升但 F1 小幅降（<=0.3pt），记录为候选方案；再用 answer_cap=5↔3 比较。

  ———

  ## 5) 交付物

  1. 代码：
      - reason/main.py 新增参数与后处理逻辑（可内嵌 helper 或新增本地函数）。
  2. 运行命令模板（简版）：
      - Baseline 与 Rerank 对照命令各一条。
  3. 结果文件：
      - 常规 *-predictions.jsonl
      - eval_result.txt 与 eval_result_corrected.txt
      - 对照表（手工/脚本）汇总 Hit@1, F1, Precision, Recall, EM, Totally Wrong, Hal Score

  ———

  ## 6) 默认假设（已选）

  - 指标优先级：Hit@1 优先。
  - 重排公式：命中数 + rank 加权。
  - 输出上限：最多 5 条答案。
  - 不改 Retriever，不新增训练，不动 Stage2。