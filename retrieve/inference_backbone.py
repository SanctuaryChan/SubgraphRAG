import argparse
import json
import os
from collections import defaultdict, deque
from statistics import mean

import torch
from tqdm import tqdm

from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import prepare_sample, set_seed


def _pair_key(u, v):
    return (u, v) if u <= v else (v, u)


def _multi_source_bfs_dist(adj, sources, max_depth):
    dist = {}
    q = deque()
    for src in sources:
        if src not in adj or src in dist:
            continue
        dist[src] = 0
        q.append(src)

    while q:
        cur = q.popleft()
        cur_d = dist[cur]
        if cur_d >= max_depth:
            continue
        for nxt in adj[cur]:
            if nxt in dist:
                continue
            dist[nxt] = cur_d + 1
            q.append(nxt)
    return dist


def _shortest_paths_limited(adj, src, dst, max_paths, hop_cap, max_preds_per_node=32):
    if src == dst:
        return [[src]]
    if (src not in adj) or (dst not in adj):
        return []

    dist = {src: 0}
    preds = defaultdict(list)
    q = deque([src])
    found_dist = None

    while q:
        cur = q.popleft()
        cur_d = dist[cur]
        if cur_d >= hop_cap:
            continue
        if (found_dist is not None) and (cur_d + 1 > found_dist):
            continue

        for nxt in adj[cur]:
            nxt_d = cur_d + 1
            if nxt_d > hop_cap:
                continue

            if nxt not in dist:
                dist[nxt] = nxt_d
                preds[nxt].append(cur)
                q.append(nxt)
                if nxt == dst:
                    found_dist = nxt_d
            elif dist[nxt] == nxt_d:
                if len(preds[nxt]) < max_preds_per_node:
                    preds[nxt].append(cur)

    if dst not in dist:
        return []
    if dist[dst] > hop_cap:
        return []

    paths = []
    cur_path = [dst]

    def _backtrack(node):
        if len(paths) >= max_paths:
            return
        if node == src:
            paths.append(list(reversed(cur_path)))
            return
        for p in preds[node]:
            cur_path.append(p)
            _backtrack(p)
            cur_path.pop()
            if len(paths) >= max_paths:
                return

    _backtrack(dst)
    return paths


def _build_backbone_first_ids(
    top_rank_ids,
    scores_cpu,
    h_id_tensor_cpu,
    t_id_tensor_cpu,
    topic_entity_ids,
    topK_reason=100,
    construct_from_top=200,
    candidate_top_m=20,
    backbone_budget=40,
    fill_budget=None,
    max_paths_per_candidate=2,
    path_hop_cap=4,
):
    total_len = len(top_rank_ids)
    reason_len = min(topK_reason, total_len)
    if reason_len == 0:
        return [], {
            "backbone_count": 0,
            "fill_count": 0,
            "num_candidates": 0,
            "num_paths": 0,
            "backbone_empty": True,
            "path_not_found": True,
            "fill_from_top200": False,
            "fill_from_global_tail": False,
        }

    if fill_budget is None:
        fill_budget = topK_reason - backbone_budget
    if fill_budget < 0:
        raise ValueError("fill_budget must be >= 0")
    if backbone_budget + fill_budget != topK_reason:
        raise ValueError("backbone_budget + fill_budget must equal topK_reason")

    backbone_budget_eff = min(backbone_budget, reason_len)
    fill_budget_eff = min(fill_budget, reason_len)

    top_reason_ids = top_rank_ids[:reason_len]
    pool_len = min(construct_from_top, total_len)
    top_pool_ids = top_rank_ids[:pool_len]

    adj = defaultdict(set)
    pair_to_best = {}
    node_score = defaultdict(float)

    for tid in top_pool_ids:
        h = int(h_id_tensor_cpu[tid].item())
        t = int(t_id_tensor_cpu[tid].item())
        s = float(scores_cpu[tid].item())
        adj[h].add(t)
        adj[t].add(h)
        node_score[h] += s
        node_score[t] += s

        key = _pair_key(h, t)
        if (key not in pair_to_best) or (s > pair_to_best[key][1]):
            pair_to_best[key] = (tid, s)

    topic_set = set(topic_entity_ids)
    topic_list = sorted(topic_set)
    dist_from_topic = _multi_source_bfs_dist(adj, topic_list, path_hop_cap)

    candidate_items = []
    for node_id, agg_score in node_score.items():
        if node_id in topic_set:
            continue
        hop = dist_from_topic.get(node_id)
        if hop is None:
            continue
        if hop == 0 or hop > path_hop_cap:
            continue
        candidate_items.append((node_id, agg_score, hop))

    # Prefer confident nodes while keeping close-by nodes slightly favored.
    candidate_items.sort(key=lambda x: (-x[1], x[2], x[0]))
    candidate_items = candidate_items[:candidate_top_m]

    path_infos = []
    for node_id, cand_score, _ in candidate_items:
        best_hop = None
        best_paths = []
        for topic_id in topic_list:
            paths = _shortest_paths_limited(
                adj=adj,
                src=topic_id,
                dst=node_id,
                max_paths=max_paths_per_candidate,
                hop_cap=path_hop_cap,
            )
            if len(paths) == 0:
                continue
            hop = len(paths[0]) - 1
            if (best_hop is None) or (hop < best_hop):
                best_hop = hop
                best_paths = paths
            elif hop == best_hop:
                remaining = max_paths_per_candidate - len(best_paths)
                if remaining > 0:
                    best_paths.extend(paths[:remaining])
            if len(best_paths) >= max_paths_per_candidate:
                break

        for node_path in best_paths[:max_paths_per_candidate]:
            triple_seq = []
            edge_scores = []
            valid = True
            for i in range(len(node_path) - 1):
                key = _pair_key(node_path[i], node_path[i + 1])
                best = pair_to_best.get(key)
                if best is None:
                    valid = False
                    break
                tid, es = best
                triple_seq.append(tid)
                edge_scores.append(es)
            if valid and len(triple_seq) > 0:
                path_infos.append(
                    {
                        "triple_seq": triple_seq,
                        "hop": len(node_path) - 1,
                        "avg_edge_score": mean(edge_scores),
                        "cand_score": cand_score,
                    }
                )

    path_infos.sort(key=lambda x: (x["hop"], -x["avg_edge_score"], -x["cand_score"]))

    backbone_ids = []
    selected_set = set()
    for info in path_infos:
        for tid in info["triple_seq"]:
            if tid in selected_set:
                continue
            backbone_ids.append(tid)
            selected_set.add(tid)
            if len(backbone_ids) >= backbone_budget_eff:
                break
        if len(backbone_ids) >= backbone_budget_eff:
            break

    fill_ids = []
    for tid in top_reason_ids:
        if tid in selected_set:
            continue
        fill_ids.append(tid)
        selected_set.add(tid)
        if len(fill_ids) >= fill_budget_eff:
            break

    final_reason_ids = list(backbone_ids) + list(fill_ids)
    fill_from_top200 = False
    fill_from_global_tail = False

    if len(final_reason_ids) < reason_len:
        for tid in top_reason_ids:
            if tid in selected_set:
                continue
            final_reason_ids.append(tid)
            selected_set.add(tid)
            if len(final_reason_ids) >= reason_len:
                break

    if len(final_reason_ids) < reason_len:
        for tid in top_pool_ids:
            if tid in selected_set:
                continue
            final_reason_ids.append(tid)
            selected_set.add(tid)
            fill_from_top200 = True
            if len(final_reason_ids) >= reason_len:
                break

    if len(final_reason_ids) < reason_len:
        for tid in top_rank_ids:
            if tid in selected_set:
                continue
            final_reason_ids.append(tid)
            selected_set.add(tid)
            fill_from_global_tail = True
            if len(final_reason_ids) >= reason_len:
                break

    if len(final_reason_ids) != reason_len:
        raise RuntimeError(
            f"Invalid top-{reason_len} assembly: got {len(final_reason_ids)} ids."
        )

    meta = {
        "backbone_count": len(backbone_ids),
        "fill_count": len(final_reason_ids) - len(backbone_ids),
        "num_candidates": len(candidate_items),
        "num_paths": len(path_infos),
        "backbone_empty": len(backbone_ids) == 0,
        "path_not_found": len(path_infos) == 0,
        "fill_from_top200": fill_from_top200,
        "fill_from_global_tail": fill_from_global_tail,
    }
    return final_reason_ids, meta


@torch.no_grad()
def main(args):
    device = torch.device("cuda:0")

    cpt = torch.load(args.path, map_location="cpu")
    config = cpt["config"]
    if args.dataset is not None:
        config = dict(config)
        config["dataset"] = dict(config["dataset"])
        config["dataset"]["name"] = args.dataset

    set_seed(config["env"]["seed"])
    torch.set_num_threads(config["env"]["num_threads"])

    infer_set = RetrieverDataset(config=config, split="test", skip_no_path=False)

    emb_size = infer_set[0]["q_emb"].shape[-1]
    model = Retriever(emb_size, **config["retriever"]).to(device)
    model.load_state_dict(cpt["model_state_dict"])
    model.eval()

    pred_dict = {}
    stat = {
        "num_samples": 0,
        "samples_backbone_empty": 0,
        "samples_path_not_found": 0,
        "samples_fill_from_top200": 0,
        "samples_fill_from_global_tail": 0,
        "sum_backbone_count": 0,
        "sum_fill_count": 0,
        "sum_candidates": 0,
        "sum_paths": 0,
    }

    for i in tqdm(range(len(infer_set))):
        raw_sample = infer_set[i]
        sample = collate_retriever([raw_sample])
        h_id_tensor_cpu, r_id_tensor_cpu, t_id_tensor_cpu = sample[0], sample[1], sample[2]
        (
            h_id_tensor,
            r_id_tensor,
            t_id_tensor,
            q_emb,
            entity_embs,
            num_non_text_entities,
            relation_embs,
            topic_entity_one_hot,
            _,
            _,
        ) = prepare_sample(device, sample)

        entity_list = raw_sample["text_entity_list"] + raw_sample["non_text_entity_list"]
        relation_list = raw_sample["relation_list"]
        top_K_triples = []
        target_relevant_triples = []

        if len(h_id_tensor) != 0:
            pred_triple_logits = model(
                h_id_tensor,
                r_id_tensor,
                t_id_tensor,
                q_emb,
                entity_embs,
                num_non_text_entities,
                relation_embs,
                topic_entity_one_hot,
            )
            pred_scores = torch.sigmoid(pred_triple_logits).reshape(-1).cpu()
            top_size = min(args.max_K, len(pred_scores))
            top_results = torch.topk(pred_scores, top_size)
            top_rank_ids = top_results.indices.tolist()

            top_reason_ids, meta = _build_backbone_first_ids(
                top_rank_ids=top_rank_ids,
                scores_cpu=pred_scores,
                h_id_tensor_cpu=h_id_tensor_cpu,
                t_id_tensor_cpu=t_id_tensor_cpu,
                topic_entity_ids=raw_sample["q_entity_id_list"],
                topK_reason=args.topK_reason,
                construct_from_top=args.construct_from_top,
                candidate_top_m=args.candidate_top_m,
                backbone_budget=args.backbone_budget,
                fill_budget=args.fill_budget,
                max_paths_per_candidate=args.max_paths_per_candidate,
                path_hop_cap=args.path_hop_cap,
            )

            used = set(top_reason_ids)
            final_rank_ids = top_reason_ids + [tid for tid in top_rank_ids if tid not in used]
            if len(final_rank_ids) != len(top_rank_ids):
                raise RuntimeError(
                    f"Rank assembly length mismatch: {len(final_rank_ids)} vs {len(top_rank_ids)}"
                )

            for triple_id in final_rank_ids:
                top_K_triples.append(
                    (
                        entity_list[h_id_tensor_cpu[triple_id].item()],
                        relation_list[r_id_tensor_cpu[triple_id].item()],
                        entity_list[t_id_tensor_cpu[triple_id].item()],
                        float(pred_scores[triple_id].item()),
                    )
                )

            stat["num_samples"] += 1
            stat["sum_backbone_count"] += meta["backbone_count"]
            stat["sum_fill_count"] += meta["fill_count"]
            stat["sum_candidates"] += meta["num_candidates"]
            stat["sum_paths"] += meta["num_paths"]
            stat["samples_backbone_empty"] += int(meta["backbone_empty"])
            stat["samples_path_not_found"] += int(meta["path_not_found"])
            stat["samples_fill_from_top200"] += int(meta["fill_from_top200"])
            stat["samples_fill_from_global_tail"] += int(meta["fill_from_global_tail"])

            target_relevant_triple_ids = (
                raw_sample["target_triple_probs"].nonzero().reshape(-1).tolist()
            )
            for triple_id in target_relevant_triple_ids:
                target_relevant_triples.append(
                    (
                        entity_list[h_id_tensor_cpu[triple_id].item()],
                        relation_list[r_id_tensor_cpu[triple_id].item()],
                        entity_list[t_id_tensor_cpu[triple_id].item()],
                    )
                )

        sample_dict = {
            "question": raw_sample["question"],
            "scored_triples": top_K_triples,
            "q_entity": raw_sample["q_entity"],
            "q_entity_in_graph": [entity_list[e_id] for e_id in raw_sample["q_entity_id_list"]],
            "a_entity": raw_sample["a_entity"],
            "a_entity_in_graph": [entity_list[e_id] for e_id in raw_sample["a_entity_id_list"]],
            "max_path_length": raw_sample["max_path_length"],
            "target_relevant_triples": target_relevant_triples,
        }
        pred_dict[raw_sample["id"]] = sample_dict

    if args.output_path is None:
        output_path = os.path.join(os.path.dirname(args.path), "retrieval_result_backbone.pth")
    else:
        output_path = args.output_path
    torch.save(pred_dict, output_path)

    if stat["num_samples"] > 0:
        summary = {
            "num_samples": stat["num_samples"],
            "avg_backbone_edges": stat["sum_backbone_count"] / stat["num_samples"],
            "avg_fill_edges": stat["sum_fill_count"] / stat["num_samples"],
            "avg_candidates": stat["sum_candidates"] / stat["num_samples"],
            "avg_paths": stat["sum_paths"] / stat["num_samples"],
            "samples_backbone_empty": stat["samples_backbone_empty"],
            "samples_path_not_found": stat["samples_path_not_found"],
            "samples_fill_from_top200": stat["samples_fill_from_top200"],
            "samples_fill_from_global_tail": stat["samples_fill_from_global_tail"],
        }
    else:
        summary = {"num_samples": 0}

    if output_path.endswith(".pth"):
        meta_path = output_path[:-4] + ".meta.json"
    else:
        meta_path = output_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved retrieval results to: {output_path}")
    print(f"Saved backbone metadata to: {meta_path}")
    print(
        "Backbone stats: "
        f"avg_backbone={summary.get('avg_backbone_edges', 0):.2f}, "
        f"avg_fill={summary.get('avg_fill_edges', 0):.2f}, "
        f"backbone_empty={summary.get('samples_backbone_empty', 0)}, "
        f"path_not_found={summary.get('samples_path_not_found', 0)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        required=True,
        help="Path to a retriever checkpoint, e.g., webqsp_xxx/cpt.pth",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        choices=["webqsp", "cwq"],
        help="Dataset override. If unset, use dataset name from checkpoint config.",
    )
    parser.add_argument(
        "--max_K",
        type=int,
        default=500,
        help="Total top-K triples to save in output.",
    )
    parser.add_argument(
        "--topK_reason",
        type=int,
        default=100,
        help="Top-K window to rewrite with backbone-first policy.",
    )
    parser.add_argument(
        "--construct_from_top",
        type=int,
        default=200,
        help="Build local graph from Stage1 top-N triples.",
    )
    parser.add_argument(
        "--candidate_top_m",
        type=int,
        default=20,
        help="Number of candidate backbone endpoint nodes.",
    )
    parser.add_argument(
        "--backbone_budget",
        type=int,
        default=40,
        help="Backbone edge budget inside topK_reason.",
    )
    parser.add_argument(
        "--fill_budget",
        type=int,
        default=None,
        help="Fill edge budget inside topK_reason. Default = topK_reason - backbone_budget.",
    )
    parser.add_argument(
        "--max_paths_per_candidate",
        type=int,
        default=2,
        help="Maximum shortest paths kept per candidate endpoint.",
    )
    parser.add_argument(
        "--path_hop_cap",
        type=int,
        default=4,
        help="Discard shortest paths longer than this hop limit.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for retrieval result. Default: <checkpoint_dir>/retrieval_result_backbone.pth",
    )
    args = parser.parse_args()

    if args.topK_reason <= 0:
        raise ValueError("topK_reason must be > 0")
    if args.backbone_budget < 0:
        raise ValueError("backbone_budget must be >= 0")
    if args.fill_budget is not None and args.fill_budget < 0:
        raise ValueError("fill_budget must be >= 0")
    if args.construct_from_top < args.topK_reason:
        raise ValueError("construct_from_top must be >= topK_reason")
    if args.max_paths_per_candidate <= 0:
        raise ValueError("max_paths_per_candidate must be > 0")
    if args.path_hop_cap <= 0:
        raise ValueError("path_hop_cap must be > 0")

    main(args)
