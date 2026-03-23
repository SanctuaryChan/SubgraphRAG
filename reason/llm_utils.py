import os
import time
import json
import hashlib
from pathlib import Path
import openai
from vllm import LLM, SamplingParams
from openai import OpenAI
from functools import partial
from prompts import icl_user_prompt, icl_ass_prompt


def _compute_rope_factor(cfg: dict, rope_scaling: dict) -> float:
    max_pos = cfg.get("max_position_embeddings")
    ori_max_pos = cfg.get("original_max_position_embeddings")
    if isinstance(max_pos, int) and isinstance(ori_max_pos, int) and ori_max_pos > 0:
        return max(1.0, float(max_pos) / float(ori_max_pos))
    # Conservative fallback. It only satisfies old vLLM validation when model
    # config misses "factor"; actual scaling behavior remains effectively unchanged.
    return 1.0


def _link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            for child in src.iterdir():
                _link_or_copy(child, dst / child.name)
        else:
            with open(src, "rb") as fr, open(dst, "wb") as fw:
                fw.write(fr.read())


def _prepare_local_model_path_for_vllm(model_name: str) -> str:
    model_path = Path(model_name)
    if not model_path.exists() or not model_path.is_dir():
        return model_name

    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        return model_name

    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    except Exception:
        return model_name

    rope_scaling = cfg.get("rope_scaling")
    if not isinstance(rope_scaling, dict) or "factor" in rope_scaling:
        return model_name

    rope_scaling = dict(rope_scaling)
    rope_scaling["factor"] = _compute_rope_factor(cfg, rope_scaling)
    cfg["rope_scaling"] = rope_scaling

    hash_key = hashlib.md5(f"{model_path.resolve()}::{json.dumps(rope_scaling, sort_keys=True)}".encode("utf-8")).hexdigest()[:12]
    patched_root = Path("/tmp/subgraphrag_model_patches")
    patched_dir = patched_root / f"{model_path.name}_{hash_key}"
    patched_cfg_path = patched_dir / "config.json"

    if not patched_cfg_path.exists():
        patched_dir.mkdir(parents=True, exist_ok=True)
        for item in model_path.iterdir():
            if item.name == "config.json":
                continue
            _link_or_copy(item, patched_dir / item.name)
        with open(patched_cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[llm_utils] Patched rope_scaling for vLLM compatibility: {patched_cfg_path}")
    else:
        print(f"[llm_utils] Using existing patched model config: {patched_cfg_path}")

    return str(patched_dir)


def llm_init(model_name, tensor_parallel_size=1, max_seq_len_to_capture=8192, max_tokens=4000, seed=0, temperature=0, frequency_penalty=0):
    if "gpt" not in model_name:
        model_for_vllm = _prepare_local_model_path_for_vllm(model_name)
        client = LLM(model=model_for_vllm, tensor_parallel_size=tensor_parallel_size, max_seq_len_to_capture=max_seq_len_to_capture)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                         frequency_penalty=frequency_penalty)
        llm = partial(client.chat, sampling_params=sampling_params, use_tqdm=False)
    else:
        # api_key = input("Enter OpenAI API key: ")
        # os.environ["OPENAI_API_KEY"] = api_key
        client = OpenAI()
        llm = partial(client.chat.completions.create, model=model_name, seed=seed, temperature=temperature, max_tokens=max_tokens)
    return llm


def get_outputs(outputs, model_name):
    if "gpt" not in model_name:
        return outputs[0].outputs[0].text
    else:
        return outputs.choices[0].message.content


def llm_inf(llm, prompts, mode, model_name):
    res = []
    if 'sys' in mode:
        conversation = [{"role": "system", "content": prompts['sys_query']}]

    if 'icl' in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    if 'sys' in mode:
        conversation.append({"role": "user", "content": prompts['user_query']})
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)

    if 'sys_cot' in mode:
        if 'clear' in mode:
            conversation = []
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts['cot_query']})
        outputs = get_outputs(llm(messages=conversation), model_name)
        res.append(outputs)
    elif "dc" in mode:
        if 'ans:' not in res[0].lower() or "ans: not available" in res[0].lower() or "ans: no information available" in res[0].lower():
            conversation.append({"role": "user", "content": prompts['cot_query']})
            outputs = get_outputs(llm(messages=conversation), model_name)
            res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res


def llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            return llm_inf(llm, each_qa, llm_mode, model_name)
        except openai.RateLimitError as e:
            wait_time = (2 ** retries) * 5  # Exponential backoff
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")


def llm_inf_all(llm, each_qa, llm_mode, model_name, max_retries=5):
    if 'gpt' in model_name:
        return llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries)
    else:
        return llm_inf(llm, each_qa, llm_mode, model_name)
