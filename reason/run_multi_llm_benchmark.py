import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def parse_alias_list(raw: str):
    if raw is None:
        return None
    aliases = [x.strip() for x in raw.split(",")]
    aliases = [x for x in aliases if x]
    return set(aliases) if aliases else None


def sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return sanitized or "model"


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (base_dir / path).resolve()


def load_model_zoo(model_zoo_path: Path):
    with open(model_zoo_path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid model zoo format: {model_zoo_path}")
    defaults = data.get("defaults", {}) or {}
    models = data.get("models", [])
    if not isinstance(models, list) or not models:
        raise ValueError(f"No models found in {model_zoo_path}")
    validated_models = []
    for model_cfg in models:
        if not isinstance(model_cfg, dict):
            raise ValueError(f"Invalid model entry in {model_zoo_path}: {model_cfg}")
        alias = model_cfg.get("alias")
        model_name = model_cfg.get("model_name")
        if not alias or not model_name:
            raise ValueError(f"Each model must define both alias and model_name, got: {model_cfg}")
        validated_models.append(model_cfg)
    return defaults, validated_models


def get_model_runtime_config(model_cfg: dict, defaults: dict, args):
    backend = model_cfg.get("backend", defaults.get("backend", "auto"))
    ollama_host = model_cfg.get("ollama_host", defaults.get("ollama_host", "http://127.0.0.1:11434"))
    tp = model_cfg.get("tensor_parallel_size", defaults.get("tensor_parallel_size", 1))
    max_seq = model_cfg.get("max_seq_len_to_capture", defaults.get("max_seq_len_to_capture", 16384))

    if args.default_tensor_parallel_size is not None:
        tp = args.default_tensor_parallel_size
    if args.default_max_seq_len_to_capture is not None:
        max_seq = args.default_max_seq_len_to_capture
    return backend, ollama_host, int(tp), int(max_seq)


def build_cmd(main_py: Path, args, model_cfg: dict, backend: str, ollama_host: str, tp: int, max_seq: int):
    cmd = [
        args.python_bin,
        str(main_py),
        "-d",
        args.dataset_name,
        "--prompt_mode",
        args.prompt_mode,
        "--llm_mode",
        args.llm_mode,
        "--llm_backend",
        backend,
        "-m",
        model_cfg["model_name"],
        "--model_alias",
        model_cfg["alias"],
        "--split",
        args.split,
        "--tensor_parallel_size",
        str(tp),
        "--max_seq_len_to_capture",
        str(max_seq),
        "--max_tokens",
        str(args.max_tokens),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--frequency_penalty",
        str(args.frequency_penalty),
        "--thres",
        str(args.thres),
    ]
    if backend == "ollama":
        cmd.extend(["--ollama_host", ollama_host])
    if args.score_dict_path is not None:
        cmd.extend(["-p", args.score_dict_path])
    if args.disable_wandb:
        cmd.append("--disable_wandb")
    return cmd


def safe_get(dct, *keys):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def summary_to_row(summary, status, error, metrics_summary_path):
    return {
        "dataset_name": summary.get("dataset_name"),
        "model_alias": summary.get("model_alias"),
        "model_name": summary.get("model_name"),
        "llm_backend": summary.get("llm_backend"),
        "ollama_host": summary.get("ollama_host"),
        "tensor_parallel_size": summary.get("tensor_parallel_size"),
        "max_seq_len_to_capture": summary.get("max_seq_len_to_capture"),
        "prompt_mode": summary.get("prompt_mode"),
        "llm_mode": summary.get("llm_mode"),
        "split": summary.get("split"),
        "max_tokens": summary.get("max_tokens"),
        "temperature": summary.get("temperature"),
        "frequency_penalty": summary.get("frequency_penalty"),
        "thres": summary.get("thres"),
        "subset_hit1": safe_get(summary, "subset_metrics", "corrected", "hit@1"),
        "subset_macro_f1": safe_get(summary, "subset_metrics", "corrected", "macro_f1"),
        "subset_exact_match": safe_get(summary, "subset_metrics", "corrected", "exact_match"),
        "subset_hal_score": safe_get(summary, "subset_metrics", "corrected", "hal_score"),
        "all_hit1": safe_get(summary, "all_metrics", "corrected", "hit@1"),
        "all_macro_f1": safe_get(summary, "all_metrics", "corrected", "macro_f1"),
        "all_exact_match": safe_get(summary, "all_metrics", "corrected", "exact_match"),
        "all_hal_score": safe_get(summary, "all_metrics", "corrected", "hal_score"),
        "all_hit_orig": safe_get(summary, "all_metrics", "original", "hit"),
        "prediction_file": summary.get("prediction_file"),
        "metrics_summary_path": str(metrics_summary_path),
        "status": status,
        "error": error,
    }


def failure_row(args, model_cfg, backend: str, ollama_host: str, tp: int, max_seq: int, status, error, metrics_summary_path):
    return {
        "dataset_name": args.dataset_name,
        "model_alias": model_cfg["alias"],
        "model_name": model_cfg["model_name"],
        "llm_backend": backend,
        "ollama_host": ollama_host if backend == "ollama" else None,
        "tensor_parallel_size": tp,
        "max_seq_len_to_capture": max_seq,
        "prompt_mode": args.prompt_mode,
        "llm_mode": args.llm_mode,
        "split": args.split,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "frequency_penalty": args.frequency_penalty,
        "thres": args.thres,
        "subset_hit1": None,
        "subset_macro_f1": None,
        "subset_exact_match": None,
        "subset_hal_score": None,
        "all_hit1": None,
        "all_macro_f1": None,
        "all_exact_match": None,
        "all_hal_score": None,
        "all_hit_orig": None,
        "prediction_file": None,
        "metrics_summary_path": str(metrics_summary_path),
        "status": status,
        "error": error,
    }


def write_leaderboard(rows, out_csv_path: Path):
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_name",
        "model_alias",
        "model_name",
        "llm_backend",
        "ollama_host",
        "tensor_parallel_size",
        "max_seq_len_to_capture",
        "prompt_mode",
        "llm_mode",
        "split",
        "max_tokens",
        "temperature",
        "frequency_penalty",
        "thres",
        "subset_hit1",
        "subset_macro_f1",
        "subset_exact_match",
        "subset_hal_score",
        "all_hit1",
        "all_macro_f1",
        "all_exact_match",
        "all_hal_score",
        "all_hit_orig",
        "prediction_file",
        "metrics_summary_path",
        "status",
        "error",
    ]
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run SubgraphRAG KGQA benchmarking on multiple LLMs.")
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Dataset name (cwq or webqsp)")
    parser.add_argument("-p", "--score_dict_path", type=str, default=None, help="Optional score_dict_path passed to main.py")
    parser.add_argument("--model_zoo", type=str, default="configs/model_zoo_local.yaml", help="Path to model zoo yaml")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model aliases to run (default: all)")

    parser.add_argument("--prompt_mode", type=str, default="scored_100")
    parser.add_argument("--llm_mode", type=str, default="sys_icl_dc")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.16)
    parser.add_argument("--thres", type=float, default=0.0)

    parser.add_argument("--default_tensor_parallel_size", type=int, default=None, help="Override TP for all models")
    parser.add_argument("--default_max_seq_len_to_capture", type=int, default=None, help="Override max_seq_len_to_capture for all models")
    parser.add_argument("--python_bin", type=str, default=sys.executable, help="Python binary used to launch main.py")
    parser.add_argument("--skip_existing", action="store_true", help="Skip model if metrics_summary.json already exists")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging in child runs")
    parser.add_argument("--dry_run", action="store_true", help="Only print commands, do not run")
    parser.add_argument("--fail_fast", action="store_true", help="Stop if one model fails")

    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    main_py = script_dir / "main.py"
    model_zoo_path = resolve_path(args.model_zoo, script_dir)
    if args.score_dict_path is not None:
        args.score_dict_path = str(resolve_path(args.score_dict_path, script_dir))

    defaults, models = load_model_zoo(model_zoo_path)
    selected_aliases = parse_alias_list(args.models)
    if selected_aliases is not None:
        models = [m for m in models if m["alias"] in selected_aliases]
        missing = selected_aliases - {m["alias"] for m in models}
        if missing:
            raise ValueError(f"Unknown aliases in --models: {sorted(missing)}")
    if not models:
        raise ValueError("No models selected to run.")

    leaderboard_rows = []
    output_root = script_dir / "results" / "KGQA" / args.dataset_name / "SubgraphRAG"

    for idx, model_cfg in enumerate(models, start=1):
        alias = model_cfg["alias"]
        alias_tag = sanitize_name(alias)
        backend, ollama_host, tp, max_seq = get_model_runtime_config(model_cfg, defaults, args)
        metrics_summary_path = output_root / alias_tag / "metrics_summary.json"
        cmd = build_cmd(main_py, args, model_cfg, backend, ollama_host, tp, max_seq)
        cmd_str = " ".join(shlex.quote(part) for part in cmd)

        print("=" * 80)
        print(f"[{idx}/{len(models)}] alias={alias} backend={backend} tp={tp} max_seq={max_seq}")
        print(f"CMD: {cmd_str}")

        if args.skip_existing and metrics_summary_path.exists():
            print(f"Skip existing: {metrics_summary_path}")
            with open(metrics_summary_path, "r") as f:
                summary = json.load(f)
            leaderboard_rows.append(summary_to_row(summary, status="skipped_existing", error="", metrics_summary_path=metrics_summary_path))
            continue

        if args.dry_run:
            leaderboard_rows.append(failure_row(args, model_cfg, backend, ollama_host, tp, max_seq, status="dry_run", error="", metrics_summary_path=metrics_summary_path))
            continue

        proc = subprocess.run(cmd, cwd=str(script_dir))
        if proc.returncode != 0:
            err_msg = f"main.py exited with code {proc.returncode}"
            print(f"ERROR: {err_msg}")
            leaderboard_rows.append(failure_row(args, model_cfg, backend, ollama_host, tp, max_seq, status="failed", error=err_msg, metrics_summary_path=metrics_summary_path))
            if args.fail_fast:
                break
            continue

        if not metrics_summary_path.exists():
            err_msg = f"Missing metrics summary: {metrics_summary_path}"
            print(f"ERROR: {err_msg}")
            leaderboard_rows.append(failure_row(args, model_cfg, backend, ollama_host, tp, max_seq, status="missing_summary", error=err_msg, metrics_summary_path=metrics_summary_path))
            if args.fail_fast:
                break
            continue

        with open(metrics_summary_path, "r") as f:
            summary = json.load(f)
        leaderboard_rows.append(summary_to_row(summary, status="ok", error="", metrics_summary_path=metrics_summary_path))

    leaderboard_path = output_root / "leaderboard.csv"
    write_leaderboard(leaderboard_rows, leaderboard_path)
    print("=" * 80)
    print(f"Saved leaderboard: {leaderboard_path}")


if __name__ == "__main__":
    main()
