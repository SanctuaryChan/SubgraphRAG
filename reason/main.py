import os
import json
import wandb
import random
import argparse
import math
import re
from tqdm import tqdm
from pathlib import Path

from preprocess.prepare_data import get_data
from preprocess.prepare_prompts import get_prompts_for_data
from llm_utils import llm_init, llm_inf_all

from metrics.evaluate_results_corrected import eval_results as eval_results_corrected
from metrics.evaluate_results import eval_results as eval_results_original


NOT_AVAILABLE_MARKERS = (
    "not available",
    "no information available",
)


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def is_not_available_answer(answer_text):
    answer_norm = normalize_text(answer_text)
    return any(marker in answer_norm for marker in NOT_AVAILABLE_MARKERS)


def parse_answer_lines(prediction_text):
    answers = []
    seen = set()
    for line in prediction_text.splitlines():
        line_strip = line.strip()
        if not line_strip.lower().startswith("ans:"):
            continue
        answer_text = line_strip[4:].strip()
        if not answer_text:
            continue
        answer_norm = normalize_text(answer_text)
        if not answer_norm or answer_norm in seen:
            continue
        seen.add(answer_norm)
        answers.append({
            "raw": answer_text,
            "norm": answer_norm,
        })
    return answers


def entity_match(answer_norm, entity_text, match_mode):
    entity_norm = normalize_text(str(entity_text))
    if not answer_norm or not entity_norm:
        return False
    if match_mode == "normalized_exact":
        return answer_norm == entity_norm
    if match_mode == "substring":
        return answer_norm in entity_norm or entity_norm in answer_norm
    raise ValueError(f"Unsupported ans_match_mode: {match_mode}")


def extract_triplet_head_tail(triplet):
    if not isinstance(triplet, (list, tuple)) or len(triplet) < 3:
        return None, None
    return triplet[0], triplet[2]


def rerank_answers_with_evidence(
    prediction_text,
    scored_triplets,
    rerank_topk,
    answer_cap,
    match_mode,
    w_count,
    w_rank,
    rank_tau,
):
    answers = parse_answer_lines(prediction_text)
    if len(answers) <= 1:
        return prediction_text

    if not scored_triplets:
        if answer_cap > 0:
            answers = answers[:answer_cap]
        return "\n".join(f"ans: {item['raw']}" for item in answers)

    topk = rerank_topk if rerank_topk and rerank_topk > 0 else len(scored_triplets)
    triplets_for_scoring = scored_triplets[:topk]
    tau = rank_tau if rank_tau > 0 else 20.0

    scored_answers = []
    has_regular_answer = any(not is_not_available_answer(item["raw"]) for item in answers)
    for original_idx, item in enumerate(answers):
        answer_norm = item["norm"]
        hit_count = 0
        rank_score = 0.0
        for rank_idx, triplet in enumerate(triplets_for_scoring, start=1):
            head, tail = extract_triplet_head_tail(triplet)
            if head is None:
                continue
            if entity_match(answer_norm, head, match_mode) or entity_match(answer_norm, tail, match_mode):
                hit_count += 1
                rank_score += math.exp(-(rank_idx - 1) / tau)

        final_score = w_count * hit_count + w_rank * rank_score
        if has_regular_answer and is_not_available_answer(item["raw"]):
            final_score = -1e9

        scored_answers.append({
            "raw": item["raw"],
            "score": final_score,
            "hit_count": hit_count,
            "idx": original_idx,
        })

    scored_answers.sort(
        key=lambda x: (x["score"], x["hit_count"], -x["idx"]),
        reverse=True,
    )

    if answer_cap > 0:
        scored_answers = scored_answers[:answer_cap]

    return "\n".join(f"ans: {item['raw']}" for item in scored_answers)


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


def main():
    parser = argparse.ArgumentParser(description="RAG for KGQA")
    parser.add_argument("-d", "--dataset_name", type=str, default="cwq", help="Dataset name")
    parser.add_argument("--prompt_mode", type=str, default="scored_100", help="Prompt mode")
    parser.add_argument("-p", "--score_dict_path", type=str)
    parser.add_argument("--llm_mode", type=str, default="sys_icl_dc", help="LLM mode")
    parser.add_argument("-m", "--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model name")
    # parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name")
    parser.add_argument("--split", type=str, default="test", help="Split")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max_seq_len_to_capture", type=int, default=8192 * 2, help="Max sequence length to capture")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Max tokens")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature")
    parser.add_argument("--frequency_penalty", type=float, default=0.16, help="Frequency penalty")
    parser.add_argument("--thres", type=float, default=0.0, help="Threshold")
    parser.add_argument("--answer_rerank", action=argparse.BooleanOptionalAction, default=True,
                        help="Re-rank ans lines by triplet evidence before saving")
    parser.add_argument("--answer_rerank_topk", type=int, default=100,
                        help="Use top-K scored triplets for answer evidence scoring")
    parser.add_argument("--answer_cap", type=int, default=5,
                        help="Keep at most this many ans lines after rerank, <=0 means no cap")
    parser.add_argument("--ans_match_mode", type=str, default="substring", choices=["substring", "normalized_exact"],
                        help="Entity matching mode for answer evidence")
    parser.add_argument("--rerank_w_count", type=float, default=1.0,
                        help="Weight for answer evidence hit count")
    parser.add_argument("--rerank_w_rank", type=float, default=1.0,
                        help="Weight for answer evidence rank score")
    parser.add_argument("--rerank_rank_tau", type=float, default=20.0,
                        help="Decay temperature for rank-based evidence score")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    prompt_mode = args.prompt_mode
    llm_mode = args.llm_mode
    model_name = args.model_name
    split = args.split
    tensor_parallel_size = args.tensor_parallel_size
    max_seq_len_to_capture = args.max_seq_len_to_capture
    max_tokens = args.max_tokens
    seed = args.seed
    temperature = args.temperature
    frequency_penalty = args.frequency_penalty
    thres = args.thres
    answer_rerank = args.answer_rerank
    answer_rerank_topk = args.answer_rerank_topk
    answer_cap = args.answer_cap
    ans_match_mode = args.ans_match_mode
    rerank_w_count = args.rerank_w_count
    rerank_w_rank = args.rerank_w_rank
    rerank_rank_tau = args.rerank_rank_tau

    rerank_tag = "ar0"
    if answer_rerank:
        rerank_tag = f"ar1-k{answer_rerank_topk}-cap{answer_cap}-m{ans_match_mode}-wc{rerank_w_count}-wr{rerank_w_rank}-tau{rerank_rank_tau}"

    pred_file_path = f"./results/KGQA/{dataset_name}/RoG/{split}/results_gen_rule_path_RoG-{dataset_name}_RoG_{split}_predictions_3_False_jsonl/predictions.jsonl"
    run_name = f"{model_name}-{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}-{rerank_tag}"
    run = wandb.init(project=f"RAG-{dataset_name}", name=run_name, config=args)

    if args.score_dict_path is None:
        if dataset_name == "webqsp":
            assert split == "test"
            score_dict_path = "./scored_triples/webqsp_240912_unidir_test.pth"
        elif dataset_name == "cwq":
            assert split == "test"
            score_dict_path = "./scored_triples/cwq_240907_unidir_test.pth"
    else:
        score_dict_path = args.score_dict_path

    raw_pred_folder_path = Path(f"./results/KGQA/{dataset_name}/SubgraphRAG/{args.model_name.split('/')[-1]}")
    raw_pred_folder_path.mkdir(parents=True, exist_ok=True)
    raw_pred_file_path = raw_pred_folder_path / f"{prompt_mode}-{llm_mode}-{frequency_penalty}-thres_{thres}-{split}-{rerank_tag}-predictions-resume.jsonl"

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

            prediction_text = res[0]
            if answer_rerank:
                prediction_text = rerank_answers_with_evidence(
                    prediction_text,
                    each_qa.get("scored_triplets", []),
                    rerank_topk=answer_rerank_topk,
                    answer_cap=answer_cap,
                    match_mode=ans_match_mode,
                    w_count=rerank_w_count,
                    w_rank=rerank_w_rank,
                    rank_tau=rerank_rank_tau,
                )

            del each_qa["graph"], each_qa["good_paths_rog"], each_qa["good_triplets_rog"], each_qa["scored_triplets"]

            each_qa["prediction"] = prediction_text
            save_checkpoint(pred_file, each_qa)

    # If the processing completes, rename the files to remove the "resume" flag
    final_pred_file_path = raw_pred_file_path.with_name(raw_pred_file_path.stem.replace("-resume", "") + raw_pred_file_path.suffix)
    os.rename(raw_pred_file_path, final_pred_file_path)
    eval_all(final_pred_file_path, run, subset=True)
    eval_all(final_pred_file_path, run, subset=False)


if __name__ == "__main__":
    main()
