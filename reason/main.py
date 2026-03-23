import os
import json
import random
import argparse
import re
from tqdm import tqdm
from pathlib import Path

try:
    import wandb
except Exception:
    wandb = None

from preprocess.prepare_data import get_data
from preprocess.prepare_prompts import get_prompts_for_data
from llm_utils import llm_init, llm_inf_all

from metrics.evaluate_results_corrected import eval_results as eval_results_corrected
from metrics.evaluate_results import eval_results as eval_results_original


class DummyRun:
    def log(self, *_args, **_kwargs):
        return None

    def finish(self):
        return None


def init_run(args, dataset_name, run_name):
    if args.disable_wandb:
        print("W&B is disabled by flag: --disable_wandb")
        return DummyRun()
    if wandb is None:
        print("W&B import failed. Falling back to disabled logging.")
        return DummyRun()
    try:
        return wandb.init(project=f"RAG-{dataset_name}", name=run_name, config=args)
    except Exception as e:
        print(f"W&B init failed ({type(e).__name__}: {e}). Falling back to disabled logging.")
        return DummyRun()


def get_defined_prompts(prompt_mode, model_name, llm_mode):
    if 'gpt' in model_name or 'gpt' in prompt_mode:
        if 'gptLabel' in prompt_mode:
            from prompts import sys_prompt_gpt, cot_prompt_gpt
            return sys_prompt_gpt, cot_prompt_gpt
        else:
            from prompts import icl_sys_prompt, icl_cot_prompt
            return icl_sys_prompt, icl_cot_prompt
    elif 'noevi' in prompt_mode:
        from prompts import noevi_sys_prompt, noevi_cot_prompt
        return noevi_sys_prompt, noevi_cot_prompt
    elif 'icl' in llm_mode:
        from prompts import icl_sys_prompt, icl_cot_prompt
        return icl_sys_prompt, icl_cot_prompt
    else:
        from prompts import sys_prompt, cot_prompt
        return sys_prompt, cot_prompt


def save_checkpoint(file_handle, data):
    file_handle.write(json.dumps(data) + "\n")


def load_checkpoint(file_path):
    if os.path.exists(file_path):
        print("*" * 50)
        print(f"Resuming from {file_path}")
        with open(file_path, "r") as f:
            ckpt = [json.loads(line) for line in f]
        try:
            print(f"Last processed item: {ckpt[-1]['id']}")
        except IndexError:
            pass
        print("*" * 50)
        return ckpt
    return []


def eval_all(pred_file_path, run, subset, split=None, eval_hops=-1):

    print("=" * 50)
    print("=" * 50)
    print(f"Evaluating on subset: {subset}")

    print("Results:")
    hit1, f1, prec, recall, em, tw, mi_f1, mi_prec, mi_recall, total_cnt, no_ans_cnt, no_ans_ratio, hal_score, stats = eval_results_corrected(str(pred_file_path), cal_f1=True, subset=subset, split=split, eval_hops=eval_hops)
    if subset:
        postfix = "_sub"
    else:
        postfix = ""
    run.log({f"results{postfix}/hit@1": hit1,
             f"results{postfix}/macro_f1": f1,
             f"results{postfix}/macro_precision": prec,
             f"results{postfix}/macro_recall": recall,
             f"results{postfix}/exact_match": em,
             f"results{postfix}/totally_wrong": tw,
             f"results{postfix}/micro_f1": mi_f1,
             f"results{postfix}/micro_precision": mi_prec,
             f"results{postfix}/micro_recall": mi_recall,
             f"results{postfix}/total_cnt": total_cnt,
             f"results{postfix}/no_ans_cnt": no_ans_cnt,
             f"results{postfix}/no_ans_ratio": no_ans_ratio,
             f"results{postfix}/hal_score": hal_score})  # score_h in the paper
    if stats is not None:
        for k, v in stats.items():
            run.log({f"stats{postfix}/{k}": v})

    hit, _, _, _ = eval_results_original(str(pred_file_path), cal_f1=True, subset=subset, eval_hops=eval_hops)
    run.log({f"results{postfix}/hit": hit})
    print("=" * 50)
    print("=" * 50)
    corrected_metrics = {
        "hit@1": hit1,
        "macro_f1": f1,
        "macro_precision": prec,
        "macro_recall": recall,
        "exact_match": em,
        "totally_wrong": tw,
        "micro_f1": mi_f1,
        "micro_precision": mi_prec,
        "micro_recall": mi_recall,
        "total_cnt": total_cnt,
        "no_ans_cnt": no_ans_cnt,
        "no_ans_ratio": no_ans_ratio,
        "hal_score": hal_score,
        "stats": stats,
    }
    original_metrics = {"hit": hit}
    return {"corrected": corrected_metrics, "original": original_metrics}


def sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return sanitized or "model"


def get_model_tag(model_name: str, model_alias: str = None) -> str:
    if model_alias:
        return sanitize_name(model_alias)
    return sanitize_name(model_name.split("/")[-1])


def main():
    parser = argparse.ArgumentParser(description="RAG for KGQA")
    parser.add_argument("-d", "--dataset_name", type=str, default="cwq", help="Dataset name")
    parser.add_argument("--prompt_mode", type=str, default="scored_100", help="Prompt mode")
    parser.add_argument("-p", "--score_dict_path", type=str)
    parser.add_argument("--llm_mode", type=str, default="sys_icl_dc", help="LLM mode")
    parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--model_alias", type=str, default=None, help="Optional alias used for output folder and run naming")
    # parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--split", type=str, default="test", help="Split")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192 * 2, help="Max sequence length to capture")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Max tokens")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature")
    parser.add_argument("--frequency_penalty", type=float, default=0.16, help="Frequency penalty")
    parser.add_argument("--thres", type=float, default=0.0, help="Threshold")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    prompt_mode = args.prompt_mode
    llm_mode = args.llm_mode
    model_name = args.model_name
    split = args.split
    model_alias = args.model_alias
    tensor_parallel_size = args.tensor_parallel_size
    max_seq_len_to_capture = args.max_seq_len_to_capture
    max_tokens = args.max_tokens
    seed = args.seed
    temperature = args.temperature
    frequency_penalty = args.frequency_penalty
    thres = args.thres

    pred_file_path = f"./results/KGQA/{dataset_name}/RoG/{split}/results_gen_rule_path_RoG-{dataset_name}_RoG_{split}_predictions_3_False_jsonl/predictions.jsonl"
    model_tag = get_model_tag(model_name, model_alias)
    run_name = sanitize_name(f"{model_tag}-{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}")
    run = init_run(args, dataset_name, run_name)

    if args.score_dict_path is None:
        if dataset_name == "webqsp":
            assert split == "test"
            score_dict_path = "./scored_triples/webqsp_240912_unidir_test.pth"
        elif dataset_name == "cwq":
            assert split == "test"
            score_dict_path = "./scored_triples/cwq_240907_unidir_test.pth"
    else:
        score_dict_path = args.score_dict_path

    raw_pred_folder_path = Path(f"./results/KGQA/{dataset_name}/SubgraphRAG/{model_tag}")
    raw_pred_folder_path.mkdir(parents=True, exist_ok=True)
    raw_pred_file_path = raw_pred_folder_path / f"{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}-predictions-resume.jsonl"

    llm = llm_init(model_name, tensor_parallel_size, max_seq_len_to_capture, max_tokens, seed, temperature, frequency_penalty)
    data = get_data(dataset_name, pred_file_path, score_dict_path, split, prompt_mode)
    sys_prompt, cot_prompt = get_defined_prompts(prompt_mode, model_name, llm_mode)
    print("Generating prompts...")
    data = get_prompts_for_data(data, prompt_mode, sys_prompt, cot_prompt, thres)

    print("Starting inference...")
    start_idx = len(load_checkpoint(raw_pred_file_path))
    with open(raw_pred_file_path, "a") as pred_file:
        for idx, each_qa in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data))):
            res = llm_inf_all(llm, each_qa, llm_mode, model_name)

            del each_qa["graph"], each_qa["good_paths_rog"], each_qa["good_triplets_rog"], each_qa["scored_triplets"]

            each_qa["prediction"] = res[0]
            save_checkpoint(pred_file, each_qa)

    # If the processing completes, rename the files to remove the "resume" flag
    final_pred_file_path = raw_pred_file_path.with_name(raw_pred_file_path.stem.replace("-resume", "") + raw_pred_file_path.suffix)
    os.rename(raw_pred_file_path, final_pred_file_path)
    subset_metrics = eval_all(final_pred_file_path, run, subset=True)
    all_metrics = eval_all(final_pred_file_path, run, subset=False)

    summary_payload = {
        "dataset_name": dataset_name,
        "split": split,
        "prompt_mode": prompt_mode,
        "llm_mode": llm_mode,
        "model_name": model_name,
        "model_alias": model_alias,
        "model_tag": model_tag,
        "tensor_parallel_size": tensor_parallel_size,
        "max_seq_len_to_capture": max_seq_len_to_capture,
        "max_tokens": max_tokens,
        "seed": seed,
        "temperature": temperature,
        "frequency_penalty": frequency_penalty,
        "thres": thres,
        "prediction_file": str(final_pred_file_path),
        "subset_metrics": subset_metrics,
        "all_metrics": all_metrics,
    }
    summary_path = raw_pred_folder_path / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Saved metrics summary: {summary_path}")
    run.finish()


if __name__ == "__main__":
    main()
