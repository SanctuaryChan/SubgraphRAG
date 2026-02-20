import os
import torch

from argparse import ArgumentParser
from tqdm import tqdm

from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.model.stage2_reranker import Stage2NodeReranker
from src.setup import set_seed, prepare_sample


def build_topk_subgraph(h_id_tensor, t_id_tensor, top_triple_ids, num_nodes, extra_node_ids=None):
    device = h_id_tensor.device
    sub_h = h_id_tensor[top_triple_ids]
    sub_t = t_id_tensor[top_triple_ids]
    node_ids = torch.unique(torch.cat([sub_h, sub_t], dim=0))

    if extra_node_ids is not None and len(extra_node_ids) > 0:
        extra_node_ids = torch.as_tensor(extra_node_ids, dtype=torch.long, device=device)
        extra_node_ids = extra_node_ids[(extra_node_ids >= 0) & (extra_node_ids < num_nodes)]
        if extra_node_ids.numel() > 0:
            node_ids = torch.unique(torch.cat([node_ids, extra_node_ids], dim=0))

    if node_ids.numel() == 0:
        return None, None

    old_to_new = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    old_to_new[node_ids] = torch.arange(node_ids.numel(), device=device)
    sub_edge_index = torch.stack([old_to_new[sub_h], old_to_new[sub_t]], dim=0)

    return node_ids, sub_edge_index


def resolve_dataset_name(config, dataset_arg):
    dataset_name = config['dataset']['name']
    if dataset_arg is None:
        return dataset_name
    if dataset_arg != dataset_name:
        raise ValueError(
            f'--dataset={dataset_arg} does not match checkpoint dataset={dataset_name}.'
        )
    return dataset_arg


@torch.no_grad()
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    stage1_cpt = torch.load(args.path, map_location='cpu')
    config = stage1_cpt['config']
    dataset_name = resolve_dataset_name(config, args.dataset)
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])

    stage2_cpt = torch.load(args.stage2_path, map_location='cpu')
    if 'dataset_name' in stage2_cpt and stage2_cpt['dataset_name'] != dataset_name:
        raise ValueError(
            f'Stage2 checkpoint dataset={stage2_cpt["dataset_name"]}, expected {dataset_name}.'
        )

    infer_set = RetrieverDataset(config=config, split='test', skip_no_path=False)
    emb_size = infer_set[0]['q_emb'].shape[-1]

    stage1_model = Retriever(emb_size, **config['retriever']).to(device)
    stage1_model.load_state_dict(stage1_cpt['model_state_dict'])
    stage1_model.eval()

    stage2_model = Stage2NodeReranker(
        in_dim=stage2_cpt['in_dim'],
        hid_dim=stage2_cpt['stage2_hparams']['hidden_dim'],
        num_layers=stage2_cpt['stage2_hparams']['num_layers'],
        dropout=stage2_cpt['stage2_hparams']['dropout'],
    ).to(device)
    stage2_model.load_state_dict(stage2_cpt['model_state_dict'])
    stage2_model.eval()

    pred_dict = dict()
    for i in tqdm(range(len(infer_set))):
        raw_sample = infer_set[i]
        sample = collate_retriever([raw_sample])
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
            num_non_text_entities, relation_embs, topic_entity_one_hot, \
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
        relation_list = raw_sample['relation_list']
        scored_triples = []
        stage1_scored_triples = []
        target_relevant_triples = []
        stage2_node_scores = []
        stage2_top_nodes = []

        if len(h_id_tensor) != 0:
            triple_logits, h_e, _, _ = stage1_model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot,
                return_aux=True
            )
            triple_scores_stage1 = torch.sigmoid(triple_logits).reshape(-1)
            k_t = min(args.max_K, triple_scores_stage1.numel())

            top_stage1 = torch.topk(triple_scores_stage1, k_t)
            top_stage1_ids = top_stage1.indices
            top_stage1_scores = top_stage1.values

            topic_ids = torch.nonzero(topic_entity_one_hot[:, 1] > 0, as_tuple=False).reshape(-1)
            node_ids, sub_edge_index = build_topk_subgraph(
                h_id_tensor, t_id_tensor, top_stage1_ids, h_e.size(0), extra_node_ids=topic_ids
            )

            node_scores_full = torch.zeros(h_e.size(0), dtype=torch.float32, device=device)
            if node_ids is not None:
                node_logits = stage2_model(h_e[node_ids], sub_edge_index)
                node_scores_sub = torch.sigmoid(node_logits)
                node_scores_full[node_ids] = node_scores_sub

                sorted_node_idx = torch.argsort(node_scores_sub, descending=True)
                for local_idx in sorted_node_idx.tolist():
                    old_id = node_ids[local_idx].item()
                    stage2_node_scores.append((
                        old_id,
                        entity_list[old_id],
                        float(node_scores_sub[local_idx].item())
                    ))

                m = min(args.node_top_m, node_scores_sub.numel())
                top_m = torch.topk(node_scores_sub, m)
                for j, local_idx in enumerate(top_m.indices.tolist()):
                    old_id = node_ids[local_idx].item()
                    stage2_top_nodes.append((
                        entity_list[old_id],
                        float(top_m.values[j].item())
                    ))

            edge_node_scores = (node_scores_full[h_id_tensor] + node_scores_full[t_id_tensor]) / 2.0
            triple_scores_final = args.alpha * triple_scores_stage1 + (1.0 - args.alpha) * edge_node_scores

            top_final = torch.topk(triple_scores_final, k_t)
            top_final_ids = top_final.indices.cpu().tolist()
            top_final_scores = top_final.values.cpu().tolist()

            for j, triple_id in enumerate(top_final_ids):
                scored_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                    top_final_scores[j]
                ))

            for j, triple_id in enumerate(top_stage1_ids.cpu().tolist()):
                stage1_scored_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                    float(top_stage1_scores[j].item())
                ))

            target_relevant_triple_ids = raw_sample['target_triple_probs'].nonzero().reshape(-1).tolist()
            for triple_id in target_relevant_triple_ids:
                target_relevant_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                ))

        sample_dict = {
            'question': raw_sample['question'],
            'scored_triples': scored_triples,
            'stage1_scored_triples': stage1_scored_triples,
            'stage2_node_scores': stage2_node_scores,
            'stage2_top_nodes': stage2_top_nodes,
            'stage2_meta': {
                'alpha': args.alpha,
                'max_K': args.max_K,
                'node_top_m': args.node_top_m,
            },
            'q_entity': raw_sample['q_entity'],
            'q_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
            'max_path_length': raw_sample['max_path_length'],
            'target_relevant_triples': target_relevant_triples,
        }

        pred_dict[raw_sample['id']] = sample_dict

    if args.output_path is None:
        root_path = os.path.dirname(os.path.abspath(args.path))
        output_path = os.path.join(root_path, 'retrieval_result_stage2.pth')
    else:
        output_path = args.output_path

    torch.save(pred_dict, output_path)
    print(f'Saved Stage2 retrieval results to: {output_path}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to Stage1 checkpoint cpt.pth')
    parser.add_argument('--stage2_path', type=str, required=True, help='Path to Stage2 checkpoint stage2_cpt.pth')
    parser.add_argument('-d', '--dataset', type=str, choices=['webqsp', 'cwq'], default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_path', type=str, default=None)

    parser.add_argument('--max_K', type=int, default=500)
    parser.add_argument('--node_top_m', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.5, help='Blend weight for Stage1 triple score')

    main(parser.parse_args())
