## SubgraphRAG DDE+Stage2 GNN（MVP）实施规格确认版

### 摘要

基于你已确认的选择，实施方案锁定为：

1. 在 retrieve 侧新增 Stage2 NodeReranker。
2. 复用 Stage1 已计算的 semantic + topic_pe + DDE 节点特征。
3. 两阶段训练：先现有 Retriever，再单独训练 Stage2（Node BCE）。
4. 推理时先 Top-K_t 选边建子图，再用 Stage2 节点分数重排三元组上下文。
5. 输出继续保留 scored_triples，并新增 node 相关字段，保证 reason 兼容。

### 已确认决策

1. 范围：先做 MVP（不做 edge gating，不做 hop 辅助头）。
2. DDE 来源：复用 Stage1 DDE（不在 Top-K 子图重算）。
3. 落地位置：放在 retrieve 模块。
4. 训练标签：先用现有弱监督（a_entity_id_list 为正类）。
5. 训练方式：两阶段训练。
6. 产物格式：保留 scored_triples 并新增字段。
7. 命令策略：新增脚本，保留旧命令。
8. 默认超参：SAGE 2层，hidden=256，TopK_t=500，NodeTopM=50，BCE(pos_weight=1)。

### 代码改动清单（按文件）

1. retrieve/src/model/retriever.py
    在 Retriever.forward（retrieve/src/model/retriever.py:102）改为返回结构化输出：
    triple_logits, node_features(h_e), edge_index, reverse_edge_index。
    保持现有三元组打分头不变，DDE 逻辑不变（retrieve/src/model/retriever.py:143）。
2. retrieve/src/model/stage2_reranker.py（新增）
    实现 Stage2NodeReranker：
    输入 x_sub, edge_index_sub，输出 node_logits。
    默认 SAGEConv 2层 + Linear(1)。
3. retrieve/train_stage2.py（新增）
    加载 Stage1 checkpoint。
    基于样本构建全图 h_e 与 edge_index，按 Stage1 triple_logits 取 Top-K_t 边形成子图。
    用 a_entity_id_list 生成 node BCE 标签训练 Stage2。
    验证集按 node recall@M / answer hit@M 早停并存 stage2_cpt.pth。
4. retrieve/inference_stage2.py（新增）
    先跑 Stage1 triple 打分。
    在 Top-K_t 子图上跑 Stage2 得到 node_scores。
    将 triple 分数重排为 triple_score_final = alpha * triple_score_stage1 + (1-alpha) * mean(node_score_h, node_score_t)。
    输出兼容旧格式并新增字段。
5. retrieve/src/dataset/retriever.py
    不改数据文件格式，仅在新脚本里读取现有字段。
    topic_entity_one_hot 与 a_entity_id_list 直接复用（见 retrieve/src/dataset/retriever.py:305 与 retrieve/src/dataset/retriever.py:303）。
6. reason 侧
    首版不改 reason 主流程。因为 reason/preprocess/prepare_data.py 只消费 scored_triples，新 retrieval_result.pth 仍可直接跑。
    后续若要消融，可增加读取 stage2_node_scores 的开关，但不作为本次 MVP 必需项。

### 公共接口与产物变更

1. 新训练命令
    python train_stage2.py -d webqsp -p <stage1_cpt_path> --top_k_t 500 --hidden_dim 256 --num_layers 2
    python train_stage2.py -d cwq -p <stage1_cpt_path> --top_k_t 500 --hidden_dim 256 --num_layers 2
2. 新推理命令
    python inference_stage2.py -p <stage1_cpt_path> --stage2_path <stage2_cpt_path> --max_K 500 --node_top_m 50 --alpha 0.5
3. retrieval_result.pth 输出 schema（兼容扩展）
    保留：scored_triples, question, a_entity, a_entity_in_graph 等。
    新增：stage1_scored_triples, stage2_node_scores, stage2_top_nodes, stage2_meta。
    scored_triples 字段写入“最终重排后”结果，确保 reason 无需改动。

### 训练与推理数据流

1. Stage1 生成 triple_logits 与 h_e。
2. 以 Top-K_t 边构图，做节点重编号 old_id -> new_id。
3. gather 子图节点特征 h_e_sub，子图边 edge_index_sub。
4. Stage2 输出 node_logits_sub，映射回全图 node_scores_full。
5. 计算最终 triple 排序并写入 scored_triples。
6. reason/main.py 按既有流程消费。

### 测试与验收场景

1. 单元测试
    Retriever.forward 输出 shape 正确。
    Top-K_t 子图重编号与回映射正确。
    stage2_node_scores 与 stage2_top_nodes 索引一致。
2. 集成测试
    inference_stage2.py 生成的 retrieval_result.pth 可被 reason/main.py 直接读取。
    webqsp 与 cwq 各跑 50 样本 smoke test 不报错。
3. 指标验收
    Retriever 侧：ans_recall@100 不低于 Stage1。
    Reason 侧：Hit@1 与 F1 相对 Stage1 基线有提升或持平（MVP 目标先验证稳定不退化）。

### 风险与规避

1. 子图过小导致答案节点被剪掉。
    默认 TopK_t=500，并记录“答案是否在子图中”的覆盖率。
2. 节点类别极不平衡。
    先用 BCE(pos_weight=1)，若正类稀疏再引入动态 pos_weight。
3. 运行开销上升。
    先固定 SAGE 2层 + hidden 256，避免首版过重。

### 假设与默认值

1. 保持现有 retrieve 数据目录与 checkpoint 结构不变。
2. 现有 Stage1 checkpoint 可稳定加载并推理。
3. reason 侧继续只依赖 scored_triples。
4. 不在本轮加入 edge gating、hop 辅助头、多任务联合训练。

• 第一阶段是否只实现文档里的 MVP（注入方式1 + Node BCE），暂不做 edge gating/辅助任务？
    answer: 只做MVP(Recommended)
• Stage2 的 DDE 你希望采用哪种来源？
    answer: 复用Stage1 DDE(Recommended)
• Stage2 输出 node score 后，先采用哪种上下文构造方式？
    answer: 重排现有triple上下文(Recommended)


• Stage2 GNN 你希望放在哪个模块里？
    answer: 放在retrieve侧(Recommended)
• Stage2 Node BCE 训练样本与标签，是否先使用现有 weak supervision（a_entity_id_list 为正，其余为负）？
    answer: 先用现有弱监督(Recommended)
• 新版推理结果文件你希望怎么兼容现有 reason？
    answer: 保留scored_triples并新增字段(Recommended)


MVP 的默认超参你希望采用哪套？
    answer: 稳妥默认(Recommended)

Stage2 训练方式你更倾向哪种？

  › 1. 两阶段训练(Recommended)  先用现有 Retriever checkpoint；再单独训练 Stage2 NodeReranker，风险最低
    2. 联合训练                 triple BCE + node BCE 同时优化，潜在更优但实现复杂
    3. 仅推理加Stage2           不训练 Stage2，仅启发式打分，创新有效性较弱
    4. None of the above        Optionally, add details in notes (tab).