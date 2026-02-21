'''
完整Retriever
结构编码+三元组打分模块
DDE三元组打分
called from: retrieve/train.py
called from: retrieve/inference.py
'''


import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing

class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, edge_index, x, edge_gate=None):
        if edge_gate is None:
            edge_gate = torch.ones(edge_index.shape[1], device=x.device, dtype=x.dtype)
        return self.propagate(edge_index, x=x, edge_gate=edge_gate)

    def message(self, x_j, edge_gate):
        return x_j * edge_gate.unsqueeze(-1)

class DDE(nn.Module):
    def __init__(
        self,
        num_rounds,
        num_reverse_rounds,
        relation_gated=False,
        gate_hidden_dim=64,
        gate_dropout=0.0,
        relation_emb_dim=None,
    ):
        super().__init__()

        # 正向传播层
        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(PEConv())
        
        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(PEConv())

        self.relation_gated = relation_gated
        if relation_gated:
            first_linear = (
                nn.LazyLinear(gate_hidden_dim)
                if relation_emb_dim is None
                else nn.Linear(relation_emb_dim, gate_hidden_dim)
            )
            self.relation_gate = nn.Sequential(
                first_linear,
                nn.ReLU(),
                nn.Dropout(gate_dropout),
                nn.Linear(gate_hidden_dim, 1),
            )
        else:
            self.relation_gate = None
    
    def forward(
        self,
        topic_entity_one_hot,
        edge_index,
        reverse_edge_index,
        r_id_tensor=None,
        relation_embs=None,
    ):
        result_list = []
        edge_gate = None
        if self.relation_gated:
            if r_id_tensor is None or relation_embs is None:
                raise ValueError(
                    'r_id_tensor and relation_embs are required when relation_gated=True.'
                )
            rel_edge_emb = relation_embs[r_id_tensor]
            edge_gate = torch.sigmoid(self.relation_gate(rel_edge_emb)).reshape(-1)

        h_pe = topic_entity_one_hot
        for layer in self.layers:
            # 每一轮PEConv都会让信号沿着 edge_index 向外走一步，因此每一轮都会产生一个新的 h_pe，这些 h_pe 就是 DDE 的多轮传播结果。
            # 如果 num_rounds=3，result_list 的前三个元素分别代表了 1-hop、2-hop、3-hop 的正向可达性特征。
            h_pe = layer(edge_index, h_pe, edge_gate=edge_gate)
            result_list.append(h_pe)
        
        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            # 反向同理
            h_pe_rev = layer(reverse_edge_index, h_pe_rev, edge_gate=edge_gate)
            result_list.append(h_pe_rev)
        
        return result_list

class Retriever(nn.Module):
    def __init__(
        self,
        emb_size,
        topic_pe,
        DDE_kwargs
    ):
        super().__init__()

        # 这些无实际语义的文本没有预训练语义特征，因此给他们一个共享的可学习的Embedding
        # 这里进行编码是为了“工程需要”，如果non-text实体在h_e中是空的（全0或不存在），那么他们在图中就像一个个“黑洞”，既无法接受邻居信息，也无法传递邻居信息
        # 同时，虽然他们没有具体语义，但是他们在图中仍然有距离和位置信息，这对于DDE的计算是非常重要的。
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(**DDE_kwargs)

        # 已知最后输出的h_triple是由h_q, h_e[h_id_tensor], h_r, h_e[t_id_tensor]拼接而成的，所以输入维度是它们的维度之和
        pred_in_size = 4 * emb_size

        # position encoding: 如果启用topic_pe，那么每个实体会有一个额外的2维位置编码（是否是topic entity），因此h_e的维度会增加2；
        # 同时DDE会输出多轮的消息传递结果，每轮都会增加2维（因为输入是topic_entity_one_hot，维度是2），因此总共增加2 * 轮数维度
        # topic_pe 最大的作用就是告诉模型：“这个节点就是起点本人。”
        if topic_pe:
            pred_in_size += 2 * 2

        # 第一个2:一个三元组包含头尾两个实体，由于这两个实体都要进入最后的拼接向量 h_triple，所所以维度是2。
        # 第二个2：代表 DDE 特征的“原始维度”，来源于 topic_entity_one_hot 的定义
        # (num_rounds + num_reverse_rounds)：代表“总传播轮数”（正向传播+反向传播）
        # 总结：pred_in_size = （头实体维度 + 尾实体维度）* one_hot向量维度 * (正向传播轮数 + 反向传播轮数)
        pred_in_size += 2 * 2 * (DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )

    def forward(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
        return_aux=False
    ):
        device = entity_embs.device

        # entity特征拼接，单独处理无语义实体
        h_e = torch.cat(
            [
                entity_embs,
                # 它不是给每个 ID 都分配一个独立的向量，而是让所有非文本实体共用同一个可学习的向量，这相当于告诉模型：“这群节点是同一类，它们都没有文本名字。”
                # 这样这样模型就不会去试图学习“ID 为 123 的实体代表什么”，而是学会了“只要是这类没有名字的节点，它们在结构上的通用贡献是什么”。这反而通过泛化降低了噪声的影响。
                self.non_text_entity_emb(
                    torch.LongTensor([0]).to(device)).expand(num_non_text_entities, -1)
            ]
        , dim=0)

        # 为实体添加“身份标签”，之前解释过topic_pe的作用是告诉模型哪个节点是问题中心，h_e中每个实体的这个标签都是0，只有问题中心的标签是1，这样模型就能区分出哪个节点是问题中心了。
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        # 正向边
        edge_index = torch.stack([
            h_id_tensor,
            t_id_tensor
        ], dim=0)

        # 反向边
        reverse_edge_index = torch.stack([
            t_id_tensor,
            h_id_tensor
        ], dim=0)

        # 调用DDE计算每个实体在图结构中相对于“问题中心”的位置
        dde_list = self.dde(
            topic_entity_one_hot,
            edge_index,
            reverse_edge_index,
            r_id_tensor=r_id_tensor,
            relation_embs=relation_embs,
        )

        # 将DDE的输出拼接到实体特征中，作为最终的结构特征输入到后续的三元组打分模块中。
        # DDE的输出是一个列表，每个元素都是一个形状为（实体数量，DDE特征维度）的张量，这些张量分别对应不同轮数的消息传递结果。
        # 我们将这些张量沿着特征维度拼接起来，形成一个新的张量，作为每个实体的结构特征。
        h_e_list.extend(dde_list)
        # Tips：之所以要先用一个列表把h_e和DDE的输出都收集起来，最后再一次性拼接，是为了避免在每轮消息传递后都进行一次拼接操作，这样会更高效一些。
        # Pytorch中的cat操作是非常低效的
        h_e = torch.cat(h_e_list, dim=1)

        h_q = q_emb
        # Potentially memory-wise problematic
        h_r = relation_embs[r_id_tensor]

        h_triple = torch.cat([
            # 维度对齐操作
            # 问题只有一个，所以它的形状可能是 (1, emb_size)，但是问题的证据三元组有多组，假设有 500 个候选三元组，它们的形状就是 (500, emb_size)
            # expand的作用就是：把那 1 行问题向量，原地复制 500 份
            h_q.expand(len(h_r), -1),
            h_e[h_id_tensor],
            h_r,
            h_e[t_id_tensor]
        ], dim=1)
        
        triple_logits = self.pred(h_triple)
        if return_aux:
            return triple_logits, h_e, edge_index, reverse_edge_index

        return triple_logits
