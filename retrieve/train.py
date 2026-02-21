'''
Retriever 主训练逻辑
'''


import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F
import wandb

from collections import defaultdict
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import BooleanOptionalAction

from src.config.retriever import load_yaml
from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample


def compute_pairwise_loss(
    pred_triple_logits,
    target_triple_probs,
    pairwise_margin,
    hard_neg_k,
    pos_cap,
    pairwise_mode,
):
    target_mask = target_triple_probs > 0
    pos_idx = torch.nonzero(target_mask, as_tuple=False).reshape(-1)
    neg_idx = torch.nonzero(~target_mask, as_tuple=False).reshape(-1)

    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        zero = pred_triple_logits.new_tensor(0.0)
        return zero, {
            'num_pairs': 0,
            'avg_pos_score': 0.0,
            'avg_hardneg_score': 0.0,
            'score_gap': 0.0,
        }

    pos_scores = pred_triple_logits[pos_idx]
    if pos_cap > 0 and pos_scores.numel() > pos_cap:
        # Hardest positives: positives with the lowest current logits.
        selected_pos = torch.topk(-pos_scores, pos_cap).indices
        pos_scores = pos_scores[selected_pos]

    neg_scores = pred_triple_logits[neg_idx]
    if hard_neg_k > 0 and neg_scores.numel() > hard_neg_k:
        selected_neg = torch.topk(neg_scores, hard_neg_k).indices
        neg_scores = neg_scores[selected_neg]

    score_diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
    if pairwise_mode == 'hinge':
        pairwise_loss = F.relu(pairwise_margin - score_diff).mean()
    elif pairwise_mode == 'logsigmoid':
        pairwise_loss = F.softplus(-(score_diff - pairwise_margin)).mean()
    else:
        raise ValueError(f'Unsupported pairwise mode: {pairwise_mode}')

    avg_pos = float(pos_scores.mean().item())
    avg_hardneg = float(neg_scores.mean().item())
    return pairwise_loss, {
        'num_pairs': int(score_diff.numel()),
        'avg_pos_score': avg_pos,
        'avg_hardneg_score': avg_hardneg,
        'score_gap': avg_pos - avg_hardneg,
    }

@torch.no_grad()
def eval_epoch(config, device, data_loader, model):
    model.eval()
    
    metric_dict = defaultdict(list)
    
    for sample in tqdm(data_loader):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        pred_triple_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot).reshape(-1)
        
        # Triple ranking
        sorted_triple_ids_pred = torch.argsort(
            pred_triple_logits, descending=True)
        triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
        triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(
            len(triple_ranks_pred), device=triple_ranks_pred.device)
        
        target_triple_ids = target_triple_probs.nonzero().squeeze(-1).to(triple_ranks_pred.device)
        num_target_triples = len(target_triple_ids)
        
        if num_target_triples == 0:
            continue

        num_total_entities = len(entity_embs) + num_non_text_entities
        for k in config['eval']['k_list']:
            recall_k_sample = (
                triple_ranks_pred[target_triple_ids] < k).sum().item()
            metric_dict[f'triple_recall@{k}'].append(
                recall_k_sample / num_target_triples)
            
            triple_mask_k = triple_ranks_pred < k
            entity_mask_k = torch.zeros(num_total_entities, device=device)
            entity_mask_k[h_id_tensor[triple_mask_k]] = 1.
            entity_mask_k[t_id_tensor[triple_mask_k]] = 1.
            a_entity_id_tensor = torch.as_tensor(a_entity_id_list, device=device, dtype=torch.long)
            if a_entity_id_tensor.numel() == 0:
                continue
            recall_k_sample_ans = entity_mask_k[a_entity_id_tensor].sum().item()
            metric_dict[f'ans_recall@{k}'].append(
                recall_k_sample_ans / len(a_entity_id_tensor))

    for key, val in metric_dict.items():
        metric_dict[key] = np.mean(val)
    
    return metric_dict

'''
训练函数
'''
def train_epoch(
    device,
    train_loader,
    model,
    optimizer,
    pairwise_weight,
    pairwise_margin,
    hard_neg_k,
    pos_cap,
    pairwise_mode,
):
    model.train()
    epoch_loss = 0.0
    epoch_bce_loss = 0.0
    epoch_pairwise_loss = 0.0
    epoch_avg_pos_score = 0.0
    epoch_avg_hardneg_score = 0.0
    epoch_score_gap = 0.0
    epoch_num_pairs = 0
    pairwise_samples = 0
    num_updates = 0

    for sample in tqdm(train_loader):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list = prepare_sample(device, sample)
            
        if len(h_id_tensor) == 0:
            continue

        pred_triple_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot)

        target_triple_probs = target_triple_probs.to(device).unsqueeze(-1)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_triple_logits, target_triple_probs)
        pairwise_loss, pairwise_stats = compute_pairwise_loss(
            pred_triple_logits.reshape(-1),
            target_triple_probs.reshape(-1),
            pairwise_margin=pairwise_margin,
            hard_neg_k=hard_neg_k,
            pos_cap=pos_cap,
            pairwise_mode=pairwise_mode,
        )
        loss = bce_loss + pairwise_weight * pairwise_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += float(loss.item())
        epoch_bce_loss += float(bce_loss.item())
        epoch_pairwise_loss += float(pairwise_loss.item())
        num_updates += 1

        if pairwise_stats['num_pairs'] > 0:
            pairwise_samples += 1
            epoch_num_pairs += pairwise_stats['num_pairs']
            epoch_avg_pos_score += pairwise_stats['avg_pos_score']
            epoch_avg_hardneg_score += pairwise_stats['avg_hardneg_score']
            epoch_score_gap += pairwise_stats['score_gap']

    if num_updates == 0:
        return {
            'loss': 0.0,
            'loss_bce': 0.0,
            'loss_pairwise': 0.0,
            'num_updates': 0,
            'pairwise_samples': 0,
            'num_pairs': 0,
            'avg_pos_score': 0.0,
            'avg_hardneg_score': 0.0,
            'score_gap': 0.0,
        }

    epoch_loss /= num_updates
    epoch_bce_loss /= num_updates
    epoch_pairwise_loss /= num_updates
    if pairwise_samples > 0:
        epoch_avg_pos_score /= pairwise_samples
        epoch_avg_hardneg_score /= pairwise_samples
        epoch_score_gap /= pairwise_samples

    log_dict = {
        'loss': epoch_loss,
        'loss_bce': epoch_bce_loss,
        'loss_pairwise': epoch_pairwise_loss,
        'num_updates': num_updates,
        'pairwise_samples': pairwise_samples,
        'num_pairs': epoch_num_pairs,
        'avg_pos_score': epoch_avg_pos_score,
        'avg_hardneg_score': epoch_avg_hardneg_score,
        'score_gap': epoch_score_gap,
    }
    return log_dict

def main(args):
    # Modify the config file for advanced settings and extensions.
    # 加载训练参数等配置文件
    config_file = f'configs/retriever/{args.dataset}.yaml'
    config = load_yaml(config_file)
    if args.relation_gated is not None:
        config['retriever']['DDE_kwargs']['relation_gated'] = args.relation_gated
    if args.gate_hidden_dim is not None:
        config['retriever']['DDE_kwargs']['gate_hidden_dim'] = args.gate_hidden_dim
    if args.gate_dropout is not None:
        config['retriever']['DDE_kwargs']['gate_dropout'] = args.gate_dropout

    config['train']['pairwise'] = {
        'weight': args.pairwise_weight,
        'margin': args.pairwise_margin,
        'hard_neg_k': args.hard_neg_k,
        'pos_cap': args.pos_cap,
        'mode': args.pairwise_mode,
    }
    
    device = torch.device('cuda:0')
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])

    # 初始化保存路径（含时间戳）以及wandb监控任务
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    config_df = pd.json_normalize(config, sep='/')
    exp_prefix = config['train']['save_prefix']
    exp_name = f'{exp_prefix}_{ts}'
    run_config = config_df.to_dict(orient='records')[0]
    run_config.update({
        'train/pairwise/weight': args.pairwise_weight,
        'train/pairwise/margin': args.pairwise_margin,
        'train/pairwise/hard_neg_k': args.hard_neg_k,
        'train/pairwise/pos_cap': args.pos_cap,
        'train/pairwise/mode': args.pairwise_mode,
        'retriever/DDE_kwargs/relation_gated': config['retriever']['DDE_kwargs'].get('relation_gated', False),
        'retriever/DDE_kwargs/gate_hidden_dim': config['retriever']['DDE_kwargs'].get('gate_hidden_dim', 64),
        'retriever/DDE_kwargs/gate_dropout': config['retriever']['DDE_kwargs'].get('gate_dropout', 0.0),
    })
    wandb.init(
        project=f'{args.dataset}',
        name=exp_name,
        config=run_config
    )
    os.makedirs(exp_name, exist_ok=True)

    # 调用RetrieverDataset加载.pkl读取数据集
    train_set = RetrieverDataset(config=config, split='train')
    val_set = RetrieverDataset(config=config, split='val')

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever)
    val_loader = DataLoader(
        val_set, batch_size=1, collate_fn=collate_retriever)
    
    emb_size = train_set[0]['q_emb'].shape[-1]

    # 加载Retriever模型
    model = Retriever(emb_size, **config['retriever']).to(device)
    optimizer = Adam(model.parameters(), **config['optimizer'])

    num_patient_epochs = 0
    best_val_metric = 0

    # 训练逻辑
    for epoch in range(config['train']['num_epochs']):
        num_patient_epochs += 1
        
        val_eval_dict = eval_epoch(config, device, val_loader, model)
        target_val_metric = val_eval_dict['triple_recall@100']
        
        if target_val_metric > best_val_metric:
            num_patient_epochs = 0
            best_val_metric = target_val_metric
            best_state_dict = {
                'config': config,
                'model_state_dict': model.state_dict()
            }
            torch.save(best_state_dict, os.path.join(exp_name, f'cpt.pth'))

            val_log = {'val/epoch': epoch}
            for key, val in val_eval_dict.items():
                val_log[f'val/{key}'] = val
            wandb.log(val_log)

        train_log_dict = train_epoch(
            device=device,
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            pairwise_weight=args.pairwise_weight,
            pairwise_margin=args.pairwise_margin,
            hard_neg_k=args.hard_neg_k,
            pos_cap=args.pos_cap,
            pairwise_mode=args.pairwise_mode,
        )
        
        train_log_dict.update({
            'num_patient_epochs': num_patient_epochs,
            'epoch': epoch
        })
        wandb.log(train_log_dict)
        if num_patient_epochs == config['train']['patience']:
            break

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('--pairwise_weight', type=float, default=0.2)
    parser.add_argument('--pairwise_margin', type=float, default=0.1)
    parser.add_argument('--hard_neg_k', type=int, default=64)
    parser.add_argument('--pos_cap', type=int, default=32)
    parser.add_argument('--pairwise_mode', type=str, default='hinge', choices=['hinge', 'logsigmoid'])
    parser.add_argument('--relation_gated', action=BooleanOptionalAction, default=None)
    parser.add_argument('--gate_hidden_dim', type=int, default=None)
    parser.add_argument('--gate_dropout', type=float, default=None)
    args = parser.parse_args()
    
    main(args)
