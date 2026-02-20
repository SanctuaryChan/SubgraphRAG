# 目标：把 DDE 注入 Stage2 的 GNN（详细设计）

你要实现的是：

1. **Stage1**（你已有）：
   DDE → 拼到实体特征 → MLP 并行给三元组打分 → Top-(K_t) 得到候选子图 (G^{(1)}_q)

2. **Stage2**（你要加）：
   在 (G^{(1)}_q) 上跑 GNN，输入包含 `semantic + topic_pe + DDE`，输出 **节点为答案的概率/得分**，再抽路径/上下文给 LLM。

---

# 一、Stage2 的输入需要什么？

Stage2 GNN 的最小输入三件套：

* `x_v`：节点特征（必须包含 DDE）
* `edge_index`：子图边
* `topic_mask / topic_one_hot`：起点锚点（用于 DDE 与可选重启机制）

### 1) 节点特征 x_v 怎么构造（推荐做法）

Stage2 的节点特征建议直接复用你 Stage1 的构造逻辑：

[
x_v = [\underbrace{e^{sem}*v}*{实体语义向量} ;|; \underbrace{pe_v}*{topic_onehot(2维)} ;|; \underbrace{dde_v}*{2\times(R+R_{rev})\text{维}}]
]

其中 `dde_v` 就是你现在 `dde_list` 里每一轮传播得到的结果拼起来。

> 关键点：
> **你不是让 GNN“自己学距离”，而是显式把“到 topic 的方向性多跳距离/可达性”作为输入特征给它。**

---

# 二、DDE 在 Stage2 要不要重新算？

我建议你从 **A** 开始做（最稳）：

## 方案 A（推荐）：在 Stage1 的“候选局部图”上算 DDE，然后同一份 DDE 给 Stage1/Stage2 共用

* Stage1 并不一定在全 KG 上打分（工程上一般会先从 topic 附近抽一个 dense subgraph 再打分）。
* 你就在这个 dense subgraph 上算 DDE（你现在这段代码就是这么干的）。
* Stage1 选 Top-(K_t) 后得到更小子图 (G^{(1)}_q)，Stage2 直接用 Stage1 算出来的 `topic_one_hot` 和 `dde_concat`（按节点 id 索引即可）。

**优点**：DDE 对距离的刻画更完整，不会因为 Top-K 裁剪而断路。
**实现最省事**：DDE 代码不用动。


---

# 三、Stage2：怎么把 DDE “注入” GNN（三种强度，从易到难）

你可以从最简单的开始，一步步加。

---

## 注入方式 1（最简单、通常也足够）：**DDE 直接拼到节点输入**

这是“最稳、最容易提分”的版本。

### 做法

1. 构造 `x = concat([entity_sem, topic_one_hot, dde_concat])`
2. 用任意 GNN（GCN/GAT/GraphSAGE/你 GNN-RAG 的那套）跑在子图上
3. 输出 node scores（答案概率）

### 代码骨架（PyG）

```python
class Stage2NodeReranker(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch_geometric.nn.SAGEConv(in_dim, hid_dim))
        for _ in range(num_layers-1):
            self.convs.append(torch_geometric.nn.SAGEConv(hid_dim, hid_dim))
        self.out = torch.nn.Linear(hid_dim, 1)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index).relu()
        return self.out(h).squeeze(-1)   # [num_nodes]
```

这已经能解决你担心的“距离缺失”——因为距离信息是显式的 `x` 的一部分。



---

# 四、Stage2 输出什么，怎么和 LLM 对接？

Stage2 输出 node score 后，你需要：

1. 取 Top-(M) 个节点作为答案候选（通常 M=20~100）
2. 从 topic 到候选节点抽路径（最短路 or top-k paths）
3. 把路径三元组列表喂给 LLM

> 你也可以保留 SubgraphRAG “Top-K triples 的子图”作为 LLM context，
> 但用 Stage2 的 node score 来决定“哪些三元组更重要”：

* 若某条边连接到高分节点/路径上，优先保留
* 否则丢弃，减少噪声 → 提升 Hit@1/F1 的稳定性

---

# 五、训练方式：Stage2 怎么训才能“确实涨点”？

你已经复现了两篇，最稳的训练策略是：

## 训练目标 1：节点答案分类（最核心）

* 正样本：gold answer entities
* 负样本：子图内其他实体（可加 hard negatives：高 degree / 语义相近实体）

loss：`BCEWithLogitsLoss` 或 `InfoNCE` 排序损失都行。

## 训练目标 2（建议加）：距离保持辅助任务（让 DDE 不被忽略）

加一个小 head 预测 hop bucket（0/1/2/3+），迫使网络利用 DDE：

* 监督信号：由 DDE 的传播轮数可以构造一个弱标签（或用 BFS 真实 hop）

这样能显著降低“网络学着学着把 DDE 当噪声丢掉”的风险。

---

# 六、把方案落到你现有 SubgraphRAG 代码里：你需要新增哪些接口？

你现在 `Retriever.forward()` 最后输出的是 `self.pred(h_triple)`（每个 triple 一个分）。

你做 hybrid 时，建议 `Retriever.forward()` 额外返回两样东西，供 Stage2 用：

1. `h_e`：拼好语义+topic_pe+DDE 的实体特征矩阵（你已经算出来了！）
2. `edge_index`：当前候选图的边（你也已经有）

也就是把：

```python
h_e = torch.cat(h_e_list, dim=1)
...
return self.pred(h_triple)
```

改成类似：

```python
triple_logits = self.pred(h_triple)           # [num_edges, 1]
return triple_logits, h_e, edge_index
```

然后在 inference 里：

* 用 `triple_logits` 取 Top-(K_t) 边得到子图
* 在子图上用 `h_e`（按节点索引裁剪/重编号）喂给 Stage2 GNN reranker

> 注意：如果 Top-K 之后你会“重编号节点”，要维护一个 `old_id -> new_id` 映射，用于从 `h_e` 里正确 gather。

---

# 七、你应该优先实现的“最小可行版本”（MVP）

为了最快得到“确实涨点”的结果，我建议你按这个顺序：

1. **Stage2 先只做：DDE 拼接到节点输入 + SAGEConv/GATConv + node BCE loss**
2. 看到 Hit@1 上升后，再加：**DDE edge gating（方式2）**
3. 再做：dynamic K / verifier / rerank-path selection（可选）

---


