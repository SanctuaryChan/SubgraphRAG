#!/usr/bin/env python3
"""Filter low-score QA samples and compare two runs sample-by-sample.

Examples
--------
Single-run hard failures (default: f1 <= 0):
  python reason/tools/filter_low_score_qa.py --pred_file <predictions.jsonl>

Single-run with stricter filter:
  python reason/tools/filter_low_score_qa.py --pred_file <predictions.jsonl> --only_hit0 --max_f1 0.2

Compare candidate vs baseline and keep regressions:
  python reason/tools/filter_low_score_qa.py \
    --pred_file <candidate_predictions.jsonl> \
    --compare_pred_file <baseline_predictions.jsonl> \
    --min_drop_f1 0.05
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def index_by_id(rows: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in rows:
        if "id" in row:
            out[str(row["id"])] = row
    return out


def infer_detailed_file(pred_file: Path) -> Path:
    name = pred_file.name
    candidates: List[Path] = []
    if "predictions.jsonl" in name:
        candidates.append(
            pred_file.with_name(
                name.replace(
                    "predictions.jsonl",
                    "full_hop-1_detailed_eval_result_corrected.jsonl",
                )
            )
        )
        candidates.append(
            pred_file.with_name(name.replace("predictions.jsonl", "detailed_eval_result.jsonl"))
        )
    else:
        candidates.append(pred_file.with_name("full_hop-1_detailed_eval_result_corrected.jsonl"))
        candidates.append(pred_file.with_name("detailed_eval_result.jsonl"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        "Could not infer detailed eval file. "
        "Pass --detailed_file explicitly. "
        f"Tried: {tried}"
    )


def as_text_prediction(record: dict) -> str:
    pred = record.get("prediction", "")
    if isinstance(pred, list):
        return "\n".join(str(x) for x in pred)
    return str(pred)


def build_single_cases(
    pred_by_id: Dict[str, dict],
    detail_by_id: Dict[str, dict],
    max_f1: float | None,
    only_hit0: bool,
    only_wrong: bool,
) -> List[dict]:
    cases: List[dict] = []
    for qid, d in detail_by_id.items():
        p = pred_by_id.get(qid, {})
        hit = float(d.get("hit", 0.0))
        f1 = float(d.get("f1", 0.0))
        precision = float(d.get("precision", d.get("precission", 0.0)))
        recall = float(d.get("recall", 0.0))

        if only_wrong:
            keep = (hit == 0.0) and (f1 == 0.0)
        else:
            keep = True
            if only_hit0:
                keep = keep and (hit == 0.0)
            if max_f1 is not None:
                keep = keep and (f1 <= max_f1)
            # default filter when user passes no thresholds
            if (not only_hit0) and (max_f1 is None):
                keep = f1 <= 0.0

        if not keep:
            continue

        cases.append(
            {
                "id": qid,
                "question": p.get("question"),
                "ground_truth": p.get("ground_truth", d.get("ground_truth", [])),
                "prediction": as_text_prediction(p) if p else d.get("prediction", []),
                "metrics": {
                    "hit": hit,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "hal_score": d.get("hal_score"),
                },
            }
        )

    cases.sort(key=lambda x: (x["metrics"]["f1"], x["metrics"]["hit"], x["metrics"]["recall"]))
    return cases


def build_compare_cases(
    cand_pred_by_id: Dict[str, dict],
    cand_detail_by_id: Dict[str, dict],
    base_pred_by_id: Dict[str, dict],
    base_detail_by_id: Dict[str, dict],
    min_drop_f1: float,
    only_hit_drop: bool,
    regression_only: bool,
) -> Tuple[List[dict], int]:
    shared_ids = sorted(set(cand_detail_by_id.keys()) & set(base_detail_by_id.keys()))
    rows: List[dict] = []

    for qid in shared_ids:
        cd = cand_detail_by_id[qid]
        bd = base_detail_by_id[qid]
        cp = cand_pred_by_id.get(qid, {})
        bp = base_pred_by_id.get(qid, {})

        cand_hit = float(cd.get("hit", 0.0))
        base_hit = float(bd.get("hit", 0.0))
        cand_f1 = float(cd.get("f1", 0.0))
        base_f1 = float(bd.get("f1", 0.0))

        delta_hit = cand_hit - base_hit
        delta_f1 = cand_f1 - base_f1

        is_regressed = (delta_hit < 0.0) or (delta_f1 < 0.0)
        if regression_only and (not is_regressed):
            continue

        if delta_f1 > (-min_drop_f1):
            if not (only_hit_drop and delta_hit < 0.0):
                continue
        if only_hit_drop and delta_hit >= 0.0:
            continue

        rows.append(
            {
                "id": qid,
                "question": cp.get("question") or bp.get("question"),
                "ground_truth": cp.get("ground_truth")
                or bp.get("ground_truth")
                or cd.get("ground_truth")
                or bd.get("ground_truth", []),
                "baseline": {
                    "metrics": {
                        "hit": base_hit,
                        "f1": base_f1,
                        "precision": float(bd.get("precision", bd.get("precission", 0.0))),
                        "recall": float(bd.get("recall", 0.0)),
                        "hal_score": bd.get("hal_score"),
                    },
                    "prediction": as_text_prediction(bp) if bp else bd.get("prediction", []),
                },
                "candidate": {
                    "metrics": {
                        "hit": cand_hit,
                        "f1": cand_f1,
                        "precision": float(cd.get("precision", cd.get("precission", 0.0))),
                        "recall": float(cd.get("recall", 0.0)),
                        "hal_score": cd.get("hal_score"),
                    },
                    "prediction": as_text_prediction(cp) if cp else cd.get("prediction", []),
                },
                "delta": {"hit": delta_hit, "f1": delta_f1},
            }
        )

    rows.sort(key=lambda x: (x["delta"]["hit"], x["delta"]["f1"], x["candidate"]["metrics"]["f1"]))
    return rows, len(shared_ids)


def save_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def preview(rows: List[dict], top_n: int, compare_mode: bool) -> None:
    print(f"Selected cases: {len(rows)}")
    for row in rows[:top_n]:
        q = (row.get("question") or "").replace("\n", " ")
        if len(q) > 120:
            q = q[:117] + "..."
        if compare_mode:
            d = row["delta"]
            print(f"id={row['id']} d_hit={d['hit']:+.0f} d_f1={d['f1']:+.4f} q={q}")
        else:
            m = row["metrics"]
            print(f"id={row['id']} hit={m['hit']:.0f} f1={m['f1']:.4f} q={q}")


def summarize_single(detail_by_id: Dict[str, dict], rows: List[dict]) -> None:
    all_f1 = [float(v.get("f1", 0.0)) for v in detail_by_id.values()]
    all_hit = [float(v.get("hit", 0.0)) for v in detail_by_id.values()]
    sel_f1 = [r["metrics"]["f1"] for r in rows] or [0.0]
    sel_hit = [r["metrics"]["hit"] for r in rows] or [0.0]
    print(f"All samples: {len(all_f1)} | mean_f1={mean(all_f1):.4f} | hit@1={mean(all_hit):.4f}")
    print(f"Selected:    {len(rows)} | mean_f1={mean(sel_f1):.4f} | hit@1={mean(sel_hit):.4f}")


def summarize_compare(shared_size: int, rows: List[dict]) -> None:
    delta_f1 = [r["delta"]["f1"] for r in rows] or [0.0]
    delta_hit = [r["delta"]["hit"] for r in rows] or [0.0]
    print(f"Shared samples: {shared_size}")
    print(
        f"Selected regressions: {len(rows)} | "
        f"mean_delta_f1={mean(delta_f1):+.4f} | "
        f"mean_delta_hit={mean(delta_hit):+.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter low-score QA cases or compare two runs case-by-case."
    )
    parser.add_argument("--pred_file", required=True, help="Candidate predictions.jsonl")
    parser.add_argument("--detailed_file", default=None, help="Candidate detailed eval jsonl")

    parser.add_argument("--compare_pred_file", default=None, help="Baseline predictions.jsonl")
    parser.add_argument("--compare_detailed_file", default=None, help="Baseline detailed eval jsonl")

    # Single-run filters
    parser.add_argument("--max_f1", type=float, default=None, help="Keep samples with f1 <= this")
    parser.add_argument("--only_hit0", action="store_true", help="Keep only hit == 0")
    parser.add_argument("--only_wrong", action="store_true", help="Keep only hit == 0 and f1 == 0")

    # Compare filters
    parser.add_argument(
        "--regression_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only regressed samples in compare mode",
    )
    parser.add_argument("--min_drop_f1", type=float, default=0.0, help="Keep delta_f1 <= -min_drop_f1")
    parser.add_argument("--only_hit_drop", action="store_true", help="Require delta_hit < 0")

    parser.add_argument("--top_n", type=int, default=50, help="Preview top-N samples in console")
    parser.add_argument("--out_file", default=None, help="Output jsonl path")
    args = parser.parse_args()

    cand_pred_file = Path(args.pred_file)
    cand_detail_file = Path(args.detailed_file) if args.detailed_file else infer_detailed_file(cand_pred_file)

    cand_pred_by_id = index_by_id(load_jsonl(cand_pred_file))
    cand_detail_by_id = index_by_id(load_jsonl(cand_detail_file))

    compare_mode = args.compare_pred_file is not None
    if compare_mode:
        base_pred_file = Path(args.compare_pred_file)
        base_detail_file = (
            Path(args.compare_detailed_file)
            if args.compare_detailed_file
            else infer_detailed_file(base_pred_file)
        )

        base_pred_by_id = index_by_id(load_jsonl(base_pred_file))
        base_detail_by_id = index_by_id(load_jsonl(base_detail_file))

        rows, shared_size = build_compare_cases(
            cand_pred_by_id,
            cand_detail_by_id,
            base_pred_by_id,
            base_detail_by_id,
            args.min_drop_f1,
            args.only_hit_drop,
            args.regression_only,
        )
        summarize_compare(shared_size, rows)
        preview(rows, args.top_n, compare_mode=True)
        out_file = (
            Path(args.out_file)
            if args.out_file
            else cand_pred_file.with_name(cand_pred_file.stem + "-regressed_cases.jsonl")
        )
    else:
        rows = build_single_cases(
            cand_pred_by_id,
            cand_detail_by_id,
            args.max_f1,
            args.only_hit0,
            args.only_wrong,
        )
        summarize_single(cand_detail_by_id, rows)
        preview(rows, args.top_n, compare_mode=False)
        out_file = (
            Path(args.out_file)
            if args.out_file
            else cand_pred_file.with_name(cand_pred_file.stem + "-low_score_cases.jsonl")
        )

    save_jsonl(out_file, rows)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
