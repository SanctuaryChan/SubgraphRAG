import argparse
import csv
import itertools
import json
import math
import os
import time

from collections import deque

import numpy as np
import torch


RANK_WEIGHT = 0.7
SEM_WEIGHT = 0.3


def safe_torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def parse_int_list(raw):
    return [int(x.strip()) for x in raw.split(',') if x.strip()]


def parse_str_list(raw):
    return [x.strip().lower() for x in raw.split(',') if x.strip()]


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    min_v = float(np.min(scores))
    max_v = float(np.max(scores))
    if max_v <= min_v:
        return np.zeros_like(scores)
    return (scores - min_v) / (max_v - min_v)


def build_undirected_adj_from_triples(triples):
    adj = {}
    for h, _, t in triples:
        if h not in adj:
            adj[h] = set()
        if t not in adj:
            adj[t] = set()
        adj[h].add(t)
        adj[t].add(h)
    return adj


def multi_source_bfs(adj, sources):
    dist = {}
    q = deque()
    for s in sources:
        if (s in adj) and (s not in dist):
            dist[s] = 0
            q.append(s)

    while q:
        cur = q.popleft()
        cur_d = dist[cur]
        for nxt in adj[cur]:
            if nxt in dist:
                continue
            dist[nxt] = cur_d + 1
            q.append(nxt)
    return dist


def has_connecting_path(triples, source_entities, target_entities):
    if (len(source_entities) == 0) or (len(target_entities) == 0):
        return False
    if len(source_entities & target_entities) > 0:
        return True

    adj = build_undirected_adj_from_triples(triples)
    visited = set()
    q = deque()
    for s in source_entities:
        if s in adj:
            visited.add(s)
            q.append(s)

    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt in target_entities:
                return True
            if nxt not in visited:
                visited.add(nxt)
                q.append(nxt)
    return False


def relation_is_type_like(relation, keywords):
    relation_l = relation.lower()
    return any(kw in relation_l for kw in keywords)


def make_sample_cache(sample_id, sample, candidate_k, rtype_keywords):
    scored_triples = sample.get('scored_triples', [])
    scored_triples = scored_triples[:candidate_k]

    cache = {
        'sample_id': sample_id,
        'valid': False,
        'triples': [],
        'sem_scores': np.array([], dtype=np.float32),
        'sem_scores_norm': np.array([], dtype=np.float32),
        'q_entities': set(sample.get('q_entity_in_graph', [])),
        'a_entities': set(sample.get('a_entity_in_graph', [])),
        'd_star': float('inf'),
        'l_values': np.array([], dtype=np.float32),
        'p_sp_indices': np.array([], dtype=np.int64),
        'type_candidates_by_entity': {},
    }

    if len(scored_triples) == 0:
        return cache
    if (len(cache['q_entities']) == 0) or (len(cache['a_entities']) == 0):
        return cache

    triples = []
    sem_scores = []
    for row in scored_triples:
        if len(row) < 4:
            continue
        h, r, t, s = row[:4]
        triples.append((h, r, t))
        try:
            sem_scores.append(float(s))
        except (TypeError, ValueError):
            sem_scores.append(0.0)

    if len(triples) == 0:
        return cache

    adj = build_undirected_adj_from_triples(triples)
    dist_s = multi_source_bfs(adj, cache['q_entities'])
    dist_a = multi_source_bfs(adj, cache['a_entities'])

    d_star = float('inf')
    for a in cache['a_entities']:
        if a in dist_s:
            d_star = min(d_star, dist_s[a])

    if not math.isfinite(d_star):
        cache['triples'] = triples
        cache['sem_scores'] = np.asarray(sem_scores, dtype=np.float32)
        cache['sem_scores_norm'] = normalize_scores(cache['sem_scores'])
        return cache

    l_values = np.full((len(triples),), np.inf, dtype=np.float32)
    for i, (h, _, t) in enumerate(triples):
        l1 = dist_s.get(h, np.inf) + 1 + dist_a.get(t, np.inf)
        l2 = dist_s.get(t, np.inf) + 1 + dist_a.get(h, np.inf)
        l_values[i] = min(l1, l2)

    p_sp_indices = np.where(l_values == d_star)[0].astype(np.int64)

    anchor_entities = set(cache['q_entities']) | set(cache['a_entities'])
    for idx in p_sp_indices:
        h, _, t = triples[int(idx)]
        anchor_entities.add(h)
        anchor_entities.add(t)

    type_candidates_by_entity = {}
    for idx, (h, r, t) in enumerate(triples):
        if not relation_is_type_like(r, rtype_keywords):
            continue
        if h in anchor_entities:
            type_candidates_by_entity.setdefault(h, []).append(idx)
        if t in anchor_entities:
            type_candidates_by_entity.setdefault(t, []).append(idx)

    sem_arr = np.asarray(sem_scores, dtype=np.float32)
    for entity, idx_list in type_candidates_by_entity.items():
        uniq = sorted(set(idx_list), key=lambda x: sem_arr[x], reverse=True)
        type_candidates_by_entity[entity] = np.asarray(uniq, dtype=np.int64)

    cache.update({
        'valid': True,
        'triples': triples,
        'sem_scores': sem_arr,
        'sem_scores_norm': normalize_scores(sem_arr),
        'd_star': float(d_star),
        'l_values': l_values,
        'p_sp_indices': p_sp_indices,
        'type_candidates_by_entity': type_candidates_by_entity,
    })
    return cache


def get_topk_indices(scores, k):
    n = len(scores)
    if n == 0:
        return np.array([], dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    part = np.argpartition(scores, -k)[-k:]
    order = np.argsort(scores[part])[::-1]
    return part[order]


def evaluate_combo(caches, delta, p_near, b_type, alpha, beta, k_eval):
    total_samples = len(caches)
    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0, 0, 0, 0

    aer_total = 0.0
    pc_total = 0.0
    rankable_count = 0
    aer_effective_count = 0
    pc_effective_count = 0

    for cache in caches:
        triples = cache['triples']
        n = len(triples)
        top_triples = []
        top_entities = set()

        if n > 0:
            rankable_count += 1
            weights = np.zeros((n,), dtype=np.float32)

            if cache['valid']:
                if len(cache['p_sp_indices']) > 0:
                    weights[cache['p_sp_indices']] = 1.0

                l_values = cache['l_values']
                near_mask = (l_values > cache['d_star']) & (l_values <= (cache['d_star'] + delta))
                near_raw = np.where(near_mask)[0]
                if near_raw.size > 0 and p_near > 0:
                    near_sorted = near_raw[np.argsort(cache['sem_scores'][near_raw])[::-1]]
                    near_selected = near_sorted[:p_near]
                    weights[near_selected] = np.maximum(weights[near_selected], alpha)

                if b_type > 0 and len(cache['type_candidates_by_entity']) > 0:
                    aux_idx_set = set()
                    for idx_arr in cache['type_candidates_by_entity'].values():
                        if len(idx_arr) == 0:
                            continue
                        for idx in idx_arr[:b_type]:
                            aux_idx_set.add(int(idx))
                    if len(aux_idx_set) > 0:
                        aux_idx = np.asarray(sorted(aux_idx_set), dtype=np.int64)
                        weights[aux_idx] = np.maximum(weights[aux_idx], beta)

            rank_scores = (RANK_WEIGHT * weights) + (SEM_WEIGHT * cache['sem_scores_norm'])
            top_idx = get_topk_indices(rank_scores, min(k_eval, n))
            top_triples = [triples[int(i)] for i in top_idx]
            for h, _, t in top_triples:
                top_entities.add(h)
                top_entities.add(t)

        a_entities = cache['a_entities']
        q_entities = cache['q_entities']

        if (n > 0) and (len(a_entities) > 0):
            aer = len(top_entities & a_entities) / len(a_entities)
            aer_effective_count += 1
        else:
            aer = 0.0

        if (n > 0) and (len(q_entities) > 0) and (len(a_entities) > 0):
            pc = 1.0 if has_connecting_path(top_triples, q_entities, a_entities) else 0.0
            pc_effective_count += 1
        else:
            pc = 0.0

        aer_total += aer
        pc_total += pc

    aer_mean = aer_total / total_samples
    pc_mean = pc_total / total_samples
    combo_score = 0.5 * aer_mean + 0.5 * pc_mean
    return aer_mean, pc_mean, combo_score, total_samples, rankable_count, aer_effective_count, pc_effective_count


def save_csv(path, rows, fieldnames):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    delta_list = parse_int_list(args.delta_list)
    pnear_list = parse_int_list(args.pnear_list)
    btype_list = parse_int_list(args.btype_list)
    rtype_keywords = parse_str_list(args.rtype_keywords)

    pred_dict = safe_torch_load(args.path)
    sample_items = list(pred_dict.items())
    if (args.max_samples is not None) and (args.max_samples > 0):
        sample_items = sample_items[:args.max_samples]

    start_t = time.time()
    caches = []
    for sample_id, sample in sample_items:
        caches.append(
            make_sample_cache(
                sample_id=sample_id,
                sample=sample,
                candidate_k=args.candidate_k,
                rtype_keywords=rtype_keywords,
            )
        )

    num_samples = len(caches)
    num_surrogate_valid_base = sum(1 for c in caches if c['valid'])

    all_results = []
    for delta, p_near, b_type in itertools.product(delta_list, pnear_list, btype_list):
        aer, pc, combo_score, num_total, num_rankable, num_aer_eff, num_pc_eff = evaluate_combo(
            caches=caches,
            delta=delta,
            p_near=p_near,
            b_type=b_type,
            alpha=args.alpha,
            beta=args.beta,
            k_eval=args.k_eval,
        )

        all_results.append({
            'dataset': args.dataset,
            'k_eval': args.k_eval,
            'candidate_k': args.candidate_k,
            'delta': delta,
            'p_near': p_near,
            'b_type': b_type,
            'alpha': args.alpha,
            'beta': args.beta,
            'num_samples': num_total,
            'num_valid_samples': num_total,
            'num_surrogate_valid_samples': num_surrogate_valid_base,
            'num_rankable_samples': num_rankable,
            'num_aer_effective_samples': num_aer_eff,
            'num_pc_effective_samples': num_pc_eff,
            'aer_at_k': round(aer, 6),
            'path_coverage_at_k': round(pc, 6),
            'score': round(combo_score, 6),
        })

    all_results_sorted = sorted(all_results, key=lambda x: x['score'], reverse=True)
    top_n = max(1, int(math.ceil(args.top_ratio * len(all_results_sorted))))
    top_subset = all_results_sorted[:top_n]
    top_configs = all_results_sorted[:min(args.top_n, len(all_results_sorted))]

    rec_delta = [int(min(r['delta'] for r in top_subset)), int(max(r['delta'] for r in top_subset))]
    rec_pnear = [int(min(r['p_near'] for r in top_subset)), int(max(r['p_near'] for r in top_subset))]
    rec_btype = [int(min(r['b_type'] for r in top_subset)), int(max(r['b_type'] for r in top_subset))]

    run_meta = {
        'dataset': args.dataset,
        'path': args.path,
        'k_eval': args.k_eval,
        'candidate_k': args.candidate_k,
        'delta_list': delta_list,
        'pnear_list': pnear_list,
        'btype_list': btype_list,
        'alpha': args.alpha,
        'beta': args.beta,
        'rtype_keywords': rtype_keywords,
        'top_ratio': args.top_ratio,
        'num_samples_total': num_samples,
        'num_surrogate_valid_base': num_surrogate_valid_base,
        'num_combinations': len(all_results),
        'elapsed_seconds': round(time.time() - start_t, 3),
        'evaluation_mode': 'all-sample-zero-fill',
    }

    rec_json = {
        'dataset': args.dataset,
        'k_eval': args.k_eval,
        'top_ratio': args.top_ratio,
        'recommended_ranges': {
            'delta': rec_delta,
            'p_near': rec_pnear,
            'b_type': rec_btype,
        },
    }

    all_results_csv = os.path.join(args.out_dir, 'all_results.csv')
    top_configs_csv = os.path.join(args.out_dir, 'top_configs.csv')
    rec_json_path = os.path.join(args.out_dir, 'recommended_range.json')
    meta_json_path = os.path.join(args.out_dir, 'run_meta.json')

    fields = [
        'dataset', 'k_eval', 'candidate_k', 'delta', 'p_near', 'b_type',
        'alpha', 'beta', 'num_samples', 'num_valid_samples',
        'num_surrogate_valid_samples', 'num_rankable_samples',
        'num_aer_effective_samples', 'num_pc_effective_samples',
        'aer_at_k', 'path_coverage_at_k', 'score'
    ]
    save_csv(all_results_csv, all_results, fields)
    save_csv(top_configs_csv, top_configs, fields)

    with open(rec_json_path, 'w', encoding='utf-8') as f:
        json.dump(rec_json, f, ensure_ascii=False, indent=2)

    with open(meta_json_path, 'w', encoding='utf-8') as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(f'Saved: {all_results_csv}')
    print(f'Saved: {top_configs_csv}')
    print(f'Saved: {rec_json_path}')
    print(f'Saved: {meta_json_path}')

    if len(top_configs) > 0:
        best = top_configs[0]
        print(
            'Best config | '
            f"delta={best['delta']} p_near={best['p_near']} b_type={best['b_type']} "
            f"score={best['score']} AER@{args.k_eval}={best['aer_at_k']} "
            f"PC@{args.k_eval}={best['path_coverage_at_k']}"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sweep surrogate-target hyperparameters (delta, p_near, b_type) '
                    'on fixed SubgraphRAG retrieval results without retraining.'
    )
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['webqsp', 'cwq'])
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to retrieval_result.pth')
    parser.add_argument('--k_eval', type=int, default=20,
                        help='Top-K used for proxy retrieval metrics (AER and Path Coverage).')
    parser.add_argument('--candidate_k', type=int, default=500,
                        help='Use at most top candidate_k scored triples per question.')
    parser.add_argument('--delta_list', type=str, default='0,1,2,3')
    parser.add_argument('--pnear_list', type=str, default='50,100,200,300,500')
    parser.add_argument('--btype_list', type=str, default='0,1,2,3,5')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--rtype_keywords', type=str,
                        default='type,types,notable,instance,category,profession,gender,nationality,ethnicity,language',
                        help='Comma-separated relation-name keywords for auxiliary type-like triples.')
    parser.add_argument('--top_ratio', type=float, default=0.15,
                        help='Ratio of top configs for recommended range extraction.')
    parser.add_argument('--top_n', type=int, default=20,
                        help='Save this many best rows to top_configs.csv.')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Optional cap of evaluated samples for quick debugging.')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory. Default: {dataset}_surrogate_sweep')

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = f'{args.dataset}_surrogate_sweep'
    main(args)
