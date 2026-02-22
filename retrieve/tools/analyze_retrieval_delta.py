#!/usr/bin/env python3
"""Compare two retrieval result .pth files and diagnose why changes do/don't help.

This script focuses on top-K evidence changes and whether those changes improve:
1) shortest-path target triple recall
2) answer-entity coverage recall
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


Triple = Tuple[str, str, str]


def to_triple_list(scored_triples: Sequence[Sequence], k: int) -> List[Triple]:
    triples: List[Triple] = []
    for row in scored_triples[:k]:
        if len(row) < 3:
            continue
        triples.append((str(row[0]), str(row[1]), str(row[2])))
    return triples


def entity_recall(topk_triples: Sequence[Triple], answer_entities: Iterable[str]) -> float | None:
    answer_set = set(str(x) for x in answer_entities)
    if not answer_set:
        return None
    ent_set = set()
    for h, _, t in topk_triples:
        ent_set.add(h)
        ent_set.add(t)
    return len(answer_set & ent_set) / len(answer_set)


def triple_recall(topk_triples: Sequence[Triple], target_triples: Iterable[Sequence]) -> float | None:
    target_set = {
        (str(x[0]), str(x[1]), str(x[2]))
        for x in target_triples
        if isinstance(x, (list, tuple)) and len(x) >= 3
    }
    if not target_set:
        return None
    topk_set = set(topk_triples)
    return len(target_set & topk_set) / len(target_set)


def mean_safe(values: List[float]) -> float:
    return mean(values) if values else 0.0


def save_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze retrieval deltas between two .pth outputs.")
    parser.add_argument("--base_pth", required=True, help="Baseline retrieval_result .pth")
    parser.add_argument("--cand_pth", required=True, help="Candidate retrieval_result .pth")
    parser.add_argument("--k", type=int, default=100, help="Top-K cutoff for analysis")
    parser.add_argument("--window_start", type=int, default=91, help="Window start rank (1-based)")
    parser.add_argument("--window_end", type=int, default=100, help="Window end rank (1-based)")
    parser.add_argument("--regression_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top_n", type=int, default=30, help="Preview top-N rows")
    parser.add_argument("--out_file", default=None, help="Output jsonl for per-sample analysis")
    args = parser.parse_args()

    base_dict: Dict = torch.load(args.base_pth, weights_only=False)
    cand_dict: Dict = torch.load(args.cand_pth, weights_only=False)
    shared_ids = sorted(set(base_dict.keys()) & set(cand_dict.keys()))

    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if not (1 <= args.window_start <= args.window_end <= args.k):
        raise ValueError("Expected 1 <= window_start <= window_end <= k")

    topk_same_ratios: List[float] = []
    window_same_ratios: List[float] = []
    target_delta_all: List[float] = []
    ans_delta_all: List[float] = []

    changed_topk = 0
    improved_target = 0
    worsened_target = 0
    unchanged_target = 0
    improved_ans = 0
    worsened_ans = 0
    unchanged_ans = 0

    changed_but_target_unchanged = 0
    promoted_target_total = 0
    demoted_target_total = 0

    rows: List[dict] = []
    ws = args.window_start - 1
    we = args.window_end

    for qid in shared_ids:
        base_sample = base_dict[qid]
        cand_sample = cand_dict[qid]

        base_topk = to_triple_list(base_sample.get("scored_triples", []), args.k)
        cand_topk = to_triple_list(cand_sample.get("scored_triples", []), args.k)

        if not base_topk or not cand_topk:
            continue

        topk_len = min(len(base_topk), len(cand_topk))
        base_topk = base_topk[:topk_len]
        cand_topk = cand_topk[:topk_len]

        base_set = set(base_topk)
        cand_set = set(cand_topk)
        inter = len(base_set & cand_set)
        same_topk_ratio = inter / max(1, topk_len)
        topk_same_ratios.append(same_topk_ratio)

        base_window = base_topk[ws:we]
        cand_window = cand_topk[ws:we]
        window_len = min(len(base_window), len(cand_window))
        if window_len > 0:
            same_window_ratio = len(set(base_window[:window_len]) & set(cand_window[:window_len])) / window_len
            window_same_ratios.append(same_window_ratio)
        else:
            same_window_ratio = 1.0

        changed = base_topk != cand_topk
        if changed:
            changed_topk += 1

        target_triples = base_sample.get("target_relevant_triples", [])
        base_target_recall = triple_recall(base_topk, target_triples)
        cand_target_recall = triple_recall(cand_topk, target_triples)
        target_delta = None
        if base_target_recall is not None and cand_target_recall is not None:
            target_delta = cand_target_recall - base_target_recall
            target_delta_all.append(target_delta)
            if target_delta > 0:
                improved_target += 1
            elif target_delta < 0:
                worsened_target += 1
            else:
                unchanged_target += 1
                if changed:
                    changed_but_target_unchanged += 1

        answer_entities = base_sample.get("a_entity_in_graph", [])
        base_ans_recall = entity_recall(base_topk, answer_entities)
        cand_ans_recall = entity_recall(cand_topk, answer_entities)
        ans_delta = None
        if base_ans_recall is not None and cand_ans_recall is not None:
            ans_delta = cand_ans_recall - base_ans_recall
            ans_delta_all.append(ans_delta)
            if ans_delta > 0:
                improved_ans += 1
            elif ans_delta < 0:
                worsened_ans += 1
            else:
                unchanged_ans += 1

        promoted = list(cand_set - base_set)
        demoted = list(base_set - cand_set)
        target_set = {
            (str(x[0]), str(x[1]), str(x[2]))
            for x in target_triples
            if isinstance(x, (list, tuple)) and len(x) >= 3
        }
        promoted_target = len(target_set & set(promoted))
        demoted_target = len(target_set & set(demoted))
        promoted_target_total += promoted_target
        demoted_target_total += demoted_target

        if args.regression_only:
            if (target_delta is None or target_delta >= 0) and (ans_delta is None or ans_delta >= 0):
                continue

        rows.append(
            {
                "id": qid,
                "question": base_sample.get("question"),
                "same_topk_ratio": same_topk_ratio,
                "same_window_ratio": same_window_ratio,
                "target_recall": {
                    "base": base_target_recall,
                    "cand": cand_target_recall,
                    "delta": target_delta,
                },
                "ans_entity_recall": {
                    "base": base_ans_recall,
                    "cand": cand_ans_recall,
                    "delta": ans_delta,
                },
                "promoted_num": len(promoted),
                "demoted_num": len(demoted),
                "promoted_target_hits": promoted_target,
                "demoted_target_hits": demoted_target,
                "promoted_examples": promoted[:5],
                "demoted_examples": demoted[:5],
            }
        )

    rows.sort(
        key=lambda x: (
            (x["target_recall"]["delta"] if x["target_recall"]["delta"] is not None else 0.0),
            (x["ans_entity_recall"]["delta"] if x["ans_entity_recall"]["delta"] is not None else 0.0),
            x["same_topk_ratio"],
        )
    )

    print("=== Retrieval Delta Summary ===")
    print(f"shared_samples={len(shared_ids)}")
    print(f"changed_topk={changed_topk} ({(changed_topk / max(1, len(shared_ids))):.2%})")
    print(f"mean_same_topk={mean_safe(topk_same_ratios):.4f}")
    print(f"mean_same_window[{args.window_start}-{args.window_end}]={mean_safe(window_same_ratios):.4f}")
    print(
        "target_recall_delta: "
        f"improved={improved_target}, worsened={worsened_target}, unchanged={unchanged_target}, "
        f"mean_delta={mean_safe(target_delta_all):+.5f}"
    )
    print(
        "ans_entity_recall_delta: "
        f"improved={improved_ans}, worsened={worsened_ans}, unchanged={unchanged_ans}, "
        f"mean_delta={mean_safe(ans_delta_all):+.5f}"
    )
    print(
        "target_hit_flow: "
        f"promoted_target_total={promoted_target_total}, demoted_target_total={demoted_target_total}, "
        f"changed_but_target_unchanged={changed_but_target_unchanged}"
    )
    print()
    print("=== Worst Cases Preview ===")
    for row in rows[: args.top_n]:
        td = row["target_recall"]["delta"]
        ad = row["ans_entity_recall"]["delta"]
        q = (row.get("question") or "").replace("\n", " ")
        if len(q) > 120:
            q = q[:117] + "..."
        print(
            f"id={row['id']} "
            f"d_target={0.0 if td is None else td:+.4f} "
            f"d_ans={0.0 if ad is None else ad:+.4f} "
            f"same@{args.k}={row['same_topk_ratio']:.3f} "
            f"same_win={row['same_window_ratio']:.3f} "
            f"q={q}"
        )

    out_file = (
        Path(args.out_file)
        if args.out_file
        else Path(args.cand_pth).with_suffix("").with_name(
            Path(args.cand_pth).stem + f".vs_{Path(args.base_pth).stem}.k{args.k}.jsonl"
        )
    )
    save_jsonl(out_file, rows)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
