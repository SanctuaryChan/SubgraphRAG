import os
import time
import torch

from argparse import ArgumentParser
from torch.optim import Adam
from torch.utils.data import DataLoader
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
        return None, None, None

    old_to_new = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    old_to_new[node_ids] = torch.arange(node_ids.numel(), device=device)
    sub_edge_index = torch.stack([old_to_new[sub_h], old_to_new[sub_t]], dim=0)

    return node_ids, sub_edge_index, old_to_new


def build_node_labels(num_nodes_sub, old_to_new, answer_ids, num_nodes_full):
    device = old_to_new.device
    labels = torch.zeros(num_nodes_sub, dtype=torch.float32, device=device)

    if len(answer_ids) == 0:
        return labels, 0, 0

    answer_ids = torch.as_tensor(answer_ids, dtype=torch.long, device=device)
    answer_ids = answer_ids[(answer_ids >= 0) & (answer_ids < num_nodes_full)]
    if answer_ids.numel() == 0:
        return labels, 0, 0

    answer_ids = torch.unique(answer_ids)
    mapped = old_to_new[answer_ids]
    mapped = mapped[mapped >= 0]
    mapped = torch.unique(mapped)
    labels[mapped] = 1.0

    return labels, int(mapped.numel()), int(answer_ids.numel())


@torch.no_grad()
def evaluate(stage1_model, stage2_model, data_loader, device, top_k_t, node_top_m):
    stage1_model.eval()
    stage2_model.eval()

    recall_list = []
    hit_list = []
    coverage_list = []

    for sample in tqdm(data_loader, leave=False):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
            num_non_text_entities, relation_embs, topic_entity_one_hot, \
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        if len(h_id_tensor) == 0:
            continue

        triple_logits, h_e, _, _ = stage1_model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot,
            return_aux=True
        )
        triple_scores = torch.sigmoid(triple_logits).reshape(-1)
        k_t = min(top_k_t, triple_scores.numel())
        if k_t == 0:
            continue

        top_triple_ids = torch.topk(triple_scores, k_t).indices
        topic_ids = torch.nonzero(topic_entity_one_hot[:, 1] > 0, as_tuple=False).reshape(-1)
        node_ids, sub_edge_index, old_to_new = build_topk_subgraph(
            h_id_tensor, t_id_tensor, top_triple_ids, h_e.size(0), extra_node_ids=topic_ids
        )
        if node_ids is None:
            continue

        labels, pos_in_sub, pos_total = build_node_labels(
            node_ids.numel(), old_to_new, a_entity_id_list, h_e.size(0)
        )
        if pos_total == 0:
            continue

        coverage_list.append(1.0 if pos_in_sub > 0 else 0.0)
        if pos_in_sub == 0:
            continue

        node_logits = stage2_model(h_e[node_ids], sub_edge_index)
        node_scores = torch.sigmoid(node_logits)

        m = min(node_top_m, node_scores.numel())
        top_node_ids = torch.topk(node_scores, m).indices
        pos_ids = labels.nonzero().reshape(-1)
        matched = torch.isin(pos_ids, top_node_ids).sum().item()

        recall_list.append(matched / max(1, pos_ids.numel()))
        hit_list.append(1.0 if matched > 0 else 0.0)

    metrics = {
        f'node_recall@{node_top_m}': sum(recall_list) / max(1, len(recall_list)),
        f'node_hit@{node_top_m}': sum(hit_list) / max(1, len(hit_list)),
        'answer_coverage_in_subgraph': sum(coverage_list) / max(1, len(coverage_list)),
        'num_eval_samples': len(recall_list),
    }
    return metrics


def train_epoch(stage1_model, stage2_model, data_loader, optimizer, criterion, device, top_k_t):
    stage1_model.eval()
    stage2_model.train()

    epoch_loss = 0.0
    num_updates = 0
    num_skipped_no_pos = 0

    for sample in tqdm(data_loader, leave=False):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
            num_non_text_entities, relation_embs, topic_entity_one_hot, \
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        if len(h_id_tensor) == 0:
            continue

        with torch.no_grad():
            triple_logits, h_e, _, _ = stage1_model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot,
                return_aux=True
            )
            triple_scores = torch.sigmoid(triple_logits).reshape(-1)

            k_t = min(top_k_t, triple_scores.numel())
            if k_t == 0:
                continue
            top_triple_ids = torch.topk(triple_scores, k_t).indices
            topic_ids = torch.nonzero(topic_entity_one_hot[:, 1] > 0, as_tuple=False).reshape(-1)
            node_ids, sub_edge_index, old_to_new = build_topk_subgraph(
                h_id_tensor, t_id_tensor, top_triple_ids, h_e.size(0), extra_node_ids=topic_ids
            )
            if node_ids is None:
                continue
            labels, pos_in_sub, _ = build_node_labels(
                node_ids.numel(), old_to_new, a_entity_id_list, h_e.size(0)
            )

        if pos_in_sub == 0:
            num_skipped_no_pos += 1
            continue

        node_logits = stage2_model(h_e[node_ids], sub_edge_index)
        loss = criterion(node_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_updates += 1

    train_log = {
        'loss': epoch_loss / max(1, num_updates),
        'num_updates': num_updates,
        'num_skipped_no_pos': num_skipped_no_pos,
    }
    return train_log


def resolve_dataset_name(config, dataset_arg):
    dataset_name = config['dataset']['name']
    if dataset_arg is None:
        return dataset_name
    if dataset_arg != dataset_name:
        raise ValueError(
            f'--dataset={dataset_arg} does not match checkpoint dataset={dataset_name}.'
        )
    return dataset_arg


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    stage1_cpt = torch.load(args.path, map_location='cpu')
    config = stage1_cpt['config']
    dataset_name = resolve_dataset_name(config, args.dataset)

    seed = config['env']['seed'] if args.seed is None else args.seed
    set_seed(seed)
    torch.set_num_threads(config['env']['num_threads'])

    train_set = RetrieverDataset(config=config, split='train')
    val_set = RetrieverDataset(config=config, split='val')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_retriever)

    emb_size = train_set[0]['q_emb'].shape[-1]
    stage1_model = Retriever(emb_size, **config['retriever']).to(device)
    stage1_model.load_state_dict(stage1_cpt['model_state_dict'])
    stage1_model.eval()
    for param in stage1_model.parameters():
        param.requires_grad = False

    with torch.no_grad():
        sample = collate_retriever([train_set[0]])
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs, \
            num_non_text_entities, relation_embs, topic_entity_one_hot, \
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)
        _, h_e, _, _ = stage1_model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot,
            return_aux=True
        )
        in_dim = h_e.size(-1)

    stage2_model = Stage2NodeReranker(
        in_dim=in_dim,
        hid_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = Adam(stage2_model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if args.save_path is None:
        root_dir = os.path.dirname(os.path.abspath(args.path))
        save_path = os.path.join(root_dir, 'stage2_cpt.pth')
    else:
        save_path = args.save_path

    best_metric = -1.0
    patient_epochs = 0

    print(f'[Stage2] dataset={dataset_name}, device={device}, in_dim={in_dim}')
    print(f'[Stage2] save_path={save_path}')

    for epoch in range(args.num_epochs):
        t0 = time.time()
        train_log = train_epoch(
            stage1_model, stage2_model, train_loader, optimizer, criterion, device, args.top_k_t
        )
        val_log = evaluate(
            stage1_model, stage2_model, val_loader, device, args.top_k_t, args.node_top_m
        )
        target_metric = val_log[f'node_hit@{args.node_top_m}']
        elapsed = time.time() - t0

        print(
            f'[Epoch {epoch:03d}] '
            f'loss={train_log["loss"]:.5f} '
            f'updates={train_log["num_updates"]} '
            f'skipped_no_pos={train_log["num_skipped_no_pos"]} '
            f'val_hit@{args.node_top_m}={val_log[f"node_hit@{args.node_top_m}"]:.4f} '
            f'val_recall@{args.node_top_m}={val_log[f"node_recall@{args.node_top_m}"]:.4f} '
            f'coverage={val_log["answer_coverage_in_subgraph"]:.4f} '
            f'time={elapsed:.1f}s'
        )

        if target_metric > best_metric:
            best_metric = target_metric
            patient_epochs = 0
            save_obj = {
                'stage1_checkpoint_path': os.path.abspath(args.path),
                'dataset_name': dataset_name,
                'stage2_hparams': {
                    'top_k_t': args.top_k_t,
                    'node_top_m': args.node_top_m,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'lr': args.lr,
                    'pos_weight': args.pos_weight,
                    'seed': seed,
                },
                'in_dim': in_dim,
                'model_state_dict': stage2_model.state_dict(),
                'stage1_config': config,
            }
            torch.save(save_obj, save_path)
            print(f'[Stage2] Saved best checkpoint to: {save_path}')
        else:
            patient_epochs += 1
            if patient_epochs >= args.patience:
                print('[Stage2] Early stop triggered.')
                break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to Stage1 checkpoint cpt.pth')
    parser.add_argument('-d', '--dataset', type=str, choices=['webqsp', 'cwq'], default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save_path', type=str, default=None)

    parser.add_argument('--top_k_t', type=int, default=500, help='Top-K triples from Stage1 to build Stage2 subgraph')
    parser.add_argument('--node_top_m', type=int, default=50, help='Validation metric cutoff M for node recall/hit')

    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pos_weight', type=float, default=1.0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)

    main(parser.parse_args())
