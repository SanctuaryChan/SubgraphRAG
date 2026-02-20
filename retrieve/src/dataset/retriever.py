'''
数据集检索工具类: 加载.plk 寻找最短路径，并将路径上的边（triple_id）标记为1，作为监督信号训练检索模型
called from: retrieve/train.py
called from: retrieve/inference.py
'''


import networkx as nx
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F

from tqdm import tqdm

class RetrieverDataset:
    def __init__(
        self,
        config,
        split,
        skip_no_path=True
    ):
        # Load pre-processed data.
        dataset_name = config['dataset']['name']
        processed_dict_list = self._load_processed(dataset_name, split)

        # Extract directed shortest paths from topic entities to answer
        # entities or vice versa as weak supervision signals for triple scoring.
        triple_score_dict = self._get_triple_scores(
            dataset_name, split, processed_dict_list)

        # Load pre-computed embeddings.
        emb_dict = self._load_emb(
            dataset_name, config['dataset']['text_encoder_name'], split)

        # Put everything together.
        self._assembly(
            processed_dict_list, triple_score_dict, emb_dict, skip_no_path)

    '''
    加载.pkl中的数据
    @return：
    - id
    - q_text
    - h_id_list
    - r_id_list
    - t_id_list
    - text_entities_list
    - non_text_entities_list
    - q_entity_id_list
    - a_entity_id_list
    '''
    def _load_processed(
        self,
        dataset_name,
        split
    ):
        processed_file = os.path.join(
            f'data_files/{dataset_name}/processed/{split}.pkl')
        with open(processed_file, 'rb') as f:
            return pickle.load(f)

    '''
    管理那些耗时的路径提取工作，并确保这些昂贵的计算结果（即作为监督信号的标签）能够被妥善保存和重复利用
    @return：
    - triple_score_dict: {
        sample_id: {
            'triple_scores': tensor, 
            'max_path_length': int
        }
    }
    '''
    def _get_triple_scores(
        self,
        dataset_name,
        split,
        processed_dict_list
    ):
        save_dir = os.path.join('data_files', dataset_name, 'triple_scores')
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'{split}.pth')

        # 缓存机制：首先检查磁盘上是否已经存在处理好的 .pth 文件。如果存在，直接加载返回，避免重复进行耗时的最短路径搜索。
        if os.path.exists(save_file):
            return torch.load(save_file)

        triple_score_dict = dict()
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            triple_scores_i, max_path_length_i = self._extract_paths_and_score(
                sample_i)

            triple_score_dict[sample_i_id] = {
                'triple_scores': triple_scores_i,
                'max_path_length': max_path_length_i
            }

        torch.save(triple_score_dict, save_file)
        
        return triple_score_dict

    '''
    在给定的候选三元组池中，找出连接“问题实体”和“答案实体”的最短路径，并将路径上的边标记为训练目标（Label=1）
    @call:
    - self._get_nx_g: 找图
    - self._shortest_path: 计算最短路径
    - self._score_triples: 将路径转为01向量
    @return：
    - triple_scores: tensor, shape = (num_triples, ), 指示每个三元组是否位于 q_entity 和 a_entity 实体之间的任何最短路径上
    - max_path_length: int, 该最短路径的跳数
    '''
    def _extract_paths_and_score(
        self,
        sample
    ):
        # 生成有向图
        nx_g = self._get_nx_g(
            sample['h_id_list'],
            sample['r_id_list'],
            sample['t_id_list']
        )

        # Each raw path is a list of entity IDs.
        # TODO: 调用 self._shortest_path() 找到所有的最短路径
        path_list_ = []
        for q_entity_id in sample['q_entity_id_list']:
            for a_entity_id in sample['a_entity_id_list']:

                # paths_q_a = [[实体ID_1, 实体ID_2, ...], [实体ID_x, 实体ID_y, ...]]
                paths_q_a = self._shortest_path(nx_g, q_entity_id, a_entity_id)
                if len(paths_q_a) > 0:
                    path_list_.extend(paths_q_a)

        if len(path_list_) == 0:
            # 标记这题没法教，因为图里没答案
            max_path_length = None
        else:
            # 初始化为0
            max_path_length = 0

        # Each processed path is a list of triple IDs.

        path_list = []
        for path in path_list_:
            # TODO: 计算这些最短路径的最大跳数
            # 这里可能会疑惑，都最短路径了，那长度应该是一样才对，但是一个“问题实例”，它可能包含多个q_entity和多个a_entity
            num_triples_path = len(path) - 1
            max_path_length = max(max_path_length, num_triples_path)
            triples_path = []

            for i in range(num_triples_path):
                h_id_i = path[i]
                t_id_i = path[i+1]
                triple_id_i_list = [
                    nx_g[h_id_i][t_id_i]['triple_id']
                ]              
                triples_path.append(triple_id_i_list)

            path_list.append(triples_path)

        num_triples = len(sample['h_id_list'])
        triple_scores = self._score_triples(
            path_list,
            num_triples
        )
        
        return triple_scores, max_path_length

    '''
    使用 networkx 库根据 .pkl 文件中的 h_id_list 和 t_id_list 构建一个有向图 DiGraph
    @retuen:
    - nx_g: networkx.DiGraph, 其中节点代表实体ID，边代表三元组（h_id, r_id, t_id），边属性包含 triple_id 和 relation_id
    '''
    def _get_nx_g(
        self,
        h_id_list,
        r_id_list,
        t_id_list
    ):
        nx_g = nx.DiGraph()
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i = h_id_list[i]
            r_i = r_id_list[i]
            t_i = t_id_list[i]
            # 将原始索引i塞入triple_id, 方便后续溯源
            nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i)

        return nx_g

    '''
    寻找双向最短路径
    @return：
    [[实体ID_1, 实体ID_2, ...], [实体ID_x, 实体ID_y, ...]]
    '''
    def _shortest_path(
        self,
        nx_g,
        q_entity_id,
        a_entity_id
    ):
        try:
            forward_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id))
        except:
            forward_paths = []
        
        try:
            backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id))
        except:
            backward_paths = []
        
        full_paths = forward_paths + backward_paths
        if (len(forward_paths) == 0) or (len(backward_paths) == 0):
            return full_paths
        
        min_path_len = min([len(path) for path in full_paths])
        refined_paths = []
        for path in full_paths:
            if len(path) == min_path_len:
                refined_paths.append(path)
        
        return refined_paths

    '''
    逐条读取候选路径，将路径上的三元组索引转为0/1向量，1表示该三元组位于 q_entity 和 a_entity 实体之间的任何最短路径上，0表示不在任何最短路径上
    @return:
    - triple_scores: tensor, shape = (num_triples, ), 指示每个三元组是否位于 q_entity 和 a_entity 实体之间的任何最短路径上
    '''
    def _score_triples(
        self,
        path_list,
        num_triples
    ):
        triple_scores = torch.zeros(num_triples)
        
        for path in path_list:
            for triple_id_list in path:
                triple_scores[triple_id_list] = 1.

        return triple_scores

    '''
    加载.pth文件
    @return：
    - emb_dict: {
        sample_id: {
            'q_emb': tensor, 
            'entity_embs': tensor, 
            'relation_embs': tensor
        }
    }
    '''
    def _load_emb(
        self,
        dataset_name,
        text_encoder_name,
        split
    ):
        file_path = f'data_files/{dataset_name}/emb/{text_encoder_name}/{split}.pth'
        dict_file = torch.load(file_path)
        
        return dict_file

    '''
    将处理好的数据、路径得分和预计算的嵌入向量整合到一个统一的数据结构中，供后续的训练和推理使用
    @return: None
    '''
    def _assembly(
        self,
        processed_dict_list,
        triple_score_dict,
        emb_dict,
        skip_no_path,
    ):
        self.processed_dict_list = []

        num_relevant_triples = []
        num_skipped = 0
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            assert sample_i_id in triple_score_dict

            triple_score_i = triple_score_dict[sample_i_id]['triple_scores']
            max_path_length_i = triple_score_dict[sample_i_id]['max_path_length']

            num_relevant_triples_i = len(triple_score_i.nonzero())
            num_relevant_triples.append(num_relevant_triples_i)

            # target_triple_probs 可以理解为是监督信号
            sample_i['target_triple_probs'] = triple_score_i
            sample_i['max_path_length'] = max_path_length_i

            if skip_no_path and (max_path_length_i in [None, 0]):
                num_skipped += 1
                continue

            sample_i.update(emb_dict[sample_i_id])

            sample_i['a_entity'] = list(set(sample_i['a_entity']))
            sample_i['a_entity_id_list'] = list(set(sample_i['a_entity_id_list']))

            # PE for topic entities.
            num_entities_i = len(sample_i['text_entity_list']) + len(sample_i['non_text_entity_list'])
            topic_entity_mask = torch.zeros(num_entities_i)
            topic_entity_mask[sample_i['q_entity_id_list']] = 1.
            topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
            sample_i['topic_entity_one_hot'] = topic_entity_one_hot.float()

            self.processed_dict_list.append(sample_i)

        median_num_relevant = int(np.median(num_relevant_triples))
        mean_num_relevant = int(np.mean(num_relevant_triples))
        max_num_relevant = int(np.max(num_relevant_triples))

        print(f'# skipped samples: {num_skipped}')
        print(f'# relevant triples | median: {median_num_relevant} | mean: {mean_num_relevant} | max: {max_num_relevant}')

    def __len__(self):
        return len(self.processed_dict_list)
    
    def __getitem__(self, i):
        return self.processed_dict_list[i]

def collate_retriever(data):
    sample = data[0]
    
    h_id_list = sample['h_id_list']
    h_id_tensor = torch.tensor(h_id_list)
    
    r_id_list = sample['r_id_list']
    r_id_tensor = torch.tensor(r_id_list)
    
    t_id_list = sample['t_id_list']
    t_id_tensor = torch.tensor(t_id_list)
    
    num_non_text_entities = len(sample['non_text_entity_list'])
    
    return h_id_tensor, r_id_tensor, t_id_tensor, sample['q_emb'],\
        sample['entity_embs'], num_non_text_entities, sample['relation_embs'],\
        sample['topic_entity_one_hot'], sample['target_triple_probs'], sample['a_entity_id_list']
