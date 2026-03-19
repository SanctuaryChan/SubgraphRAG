from collections import deque

import numpy as np
import pandas as pd
import torch


def build_undirected_adj(triples):
    adj = dict()
    for h, _, t in triples:
        if h not in adj:
            adj[h] = set()
        if t not in adj:
            adj[t] = set()
        adj[h].add(t)
        adj[t].add(h)
    return adj


def has_connected_path(adj, source_entities, target_entities):
    if (len(source_entities) == 0) or (len(target_entities) == 0):
        return False

    target_entities = set(target_entities)
    visited = set()
    q = deque()

    for source in source_entities:
        if source in target_entities:
            return True
        if source in adj:
            visited.add(source)
            q.append(source)

    while len(q) > 0:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt in target_entities:
                return True
            if nxt not in visited:
                visited.add(nxt)
                q.append(nxt)

    return False


def main(args):
    pred_dict = torch.load(args.path)
    gpt_triple_dict = torch.load(f'data_files/{args.dataset}/gpt_triples.pth')
    k_list = [int(k) for k in args.k_list.split(',')]
    
    metric_dict = dict()
    for k in k_list:
        metric_dict[f'ans_recall@{k}'] = []
        metric_dict[f'shortest_path_triple_recall@{k}'] = []
        metric_dict[f'gpt_triple_recall@{k}'] = []
        metric_dict[f'path_coverage@{k}'] = []
    
    for sample_id in pred_dict:
        scored_triples = pred_dict[sample_id]['scored_triples']
        q_entity_in_graph = set(pred_dict[sample_id]['q_entity_in_graph'])
        a_entity_in_graph = set(pred_dict[sample_id]['a_entity_in_graph'])

        for k in k_list:
            triples_k = [(h, r, t) for h, r, t, _ in scored_triples[:k]]
            graph_k = build_undirected_adj(triples_k)
            if has_connected_path(graph_k, q_entity_in_graph, a_entity_in_graph):
                metric_dict[f'path_coverage@{k}'].append(1.0)
            else:
                metric_dict[f'path_coverage@{k}'].append(0.0)

        if len(scored_triples) == 0:
            continue
        
        h_list, r_list, t_list, _ = zip(*scored_triples)
        
        if len(a_entity_in_graph) > 0:
            for k in k_list:
                entities_k = set(h_list[:k] + t_list[:k])
                metric_dict[f'ans_recall@{k}'].append(
                    len(a_entity_in_graph & entities_k) / len(a_entity_in_graph)
                )
        
        triples = list(zip(h_list, r_list, t_list))
        shortest_path_triples = set(pred_dict[sample_id]['target_relevant_triples'])
        if len(shortest_path_triples) > 0:
            for k in k_list:
                triples_k = set(triples[:k])
                metric_dict[f'shortest_path_triple_recall@{k}'].append(
                    len(shortest_path_triples & triples_k) / len(shortest_path_triples)
                )
        
        gpt_triples = set(gpt_triple_dict.get(sample_id, []))
        if len(gpt_triples) > 0:
            for k in k_list:
                triples_k = set(triples[:k])
                metric_dict[f'gpt_triple_recall@{k}'].append(
                    len(gpt_triples & triples_k) / len(gpt_triples)
                )

    for metric, val in metric_dict.items():
        metric_dict[metric] = np.mean(val)
    
    table_dict = {
        'K': k_list,
        'ans_recall': [
            round(metric_dict[f'ans_recall@{k}'], 3) for k in k_list
        ],
        'shortest_path_triple_recall': [
            round(metric_dict[f'shortest_path_triple_recall@{k}'], 3) for k in k_list
        ],
        'gpt_triple_recall': [
            round(metric_dict[f'gpt_triple_recall@{k}'], 3) for k in k_list
        ],
        'path_coverage': [
            round(metric_dict[f'path_coverage@{k}'], 3) for k in k_list
        ]
    }
    df = pd.DataFrame(table_dict)
    print(df.to_string(index=False))

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to retrieval result')
    parser.add_argument('--k_list', type=str, default='50,100,200,400',
                        help='Comma-separated list of K values for top-K recall evaluation')
    args = parser.parse_args()
    
    main(args)
