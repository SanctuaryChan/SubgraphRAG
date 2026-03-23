import os
import time
import json
import hashlib
from pathlib import Path
from functools import partial

import openai
from openai import OpenAI

try:
    from vllm import LLM, SamplingParams
    _VLLM_IMPORT_ERROR = None
except Exception as e:
    LLM = None
    SamplingParams = None
    _VLLM_IMPORT_ERROR = e

from prompts import icl_user_prompt, icl_ass_prompt


VALID_LLM_BACKENDS = {"auto", "local_vllm", "openai", "openai_compatible"}


def normalize_llm_backend(llm_backend: str, model_name: str) -> str:
    backend = (llm_backend or "auto").strip().lower()
    if backend not in VALID_LLM_BACKENDS:
        raise ValueError(f"Unsupported llm backend: {llm_backend}. Expected one of {sorted(VALID_LLM_BACKENDS)}")
    if backend == "auto":
        return "openai" if "gpt" in model_name else "local_vllm"
    return backend


def _compute_rope_factor(context_cfg: dict, rope_scaling: dict, root_cfg: dict) -> float:
    max_pos = context_cfg.get("max_position_embeddings")
    if not isinstance(max_pos, int):
        max_pos = root_cfg.get("max_position_embeddings")

    ori_max_pos = rope_scaling.get("original_max_position_embeddings")
    if not isinstance(ori_max_pos, int):
        ori_max_pos = context_cfg.get("original_max_position_embeddings")
    if not isinstance(ori_max_pos, int):
        ori_max_pos = root_cfg.get("original_max_position_embeddings")

    if isinstance(max_pos, int) and isinstance(ori_max_pos, int) and ori_max_pos > 0:
        return max(1.0, float(max_pos) / float(ori_max_pos))
    # Conservative fallback. It only satisfies old vLLM validation when model
    # config misses "factor"; actual scaling behavior remains effectively unchanged.
    return 1.0


def _has_valid_rope_factor(rope_scaling: dict) -> bool:
    factor = rope_scaling.get("factor")
    return isinstance(factor, (int, float)) and factor > 0


def _patch_rope_scaling_in_cfg(node, root_cfg: dict, path: str = "config"):
    patched_paths = []

    if isinstance(node, dict):
        rope_scaling = node.get("rope_scaling")
        if isinstance(rope_scaling, dict) and not _has_valid_rope_factor(rope_scaling):
            patched_rope_scaling = dict(rope_scaling)
            patched_rope_scaling["factor"] = _compute_rope_factor(node, patched_rope_scaling, root_cfg)
            node["rope_scaling"] = patched_rope_scaling
            patched_paths.append(f"{path}.rope_scaling")

        for key, value in node.items():
            patched_paths.extend(_patch_rope_scaling_in_cfg(value, root_cfg, f"{path}.{key}"))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            patched_paths.extend(_patch_rope_scaling_in_cfg(value, root_cfg, f"{path}[{idx}]"))

    return patched_paths


def _contains_rope_scaling(node) -> bool:
    if isinstance(node, dict):
        if isinstance(node.get("rope_scaling"), dict):
            return True
        return any(_contains_rope_scaling(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_rope_scaling(value) for value in node)
    return False


def _raise_on_known_incompatible_config(cfg: dict, model_name: str):
    architectures = cfg.get("architectures") or []
    text_cfg = cfg.get("text_config")
    has_rope_parameters = isinstance(text_cfg, dict) and isinstance(text_cfg.get("rope_parameters"), dict)
    is_qwen35_multimodal = "Qwen3_5ForConditionalGeneration" in architectures or (
        cfg.get("model_type") == "qwen3_5" and isinstance(cfg.get("vision_config"), dict)
    )

    if is_qwen35_multimodal and has_rope_parameters and not _contains_rope_scaling(cfg):
        raise RuntimeError(
            "Detected a Qwen3.5 multimodal checkpoint at "
            f"{model_name} with text_config.rope_parameters but no rope_scaling. "
            "Direct local_vllm loading in this repository is not suitable for this "
            "checkpoint format. Start a newer vLLM OpenAI-compatible server with "
            "--language-model-only and use llm_backend=openai_compatible instead."
        )


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

    _raise_on_known_incompatible_config(cfg, model_name)

    patched_paths = _patch_rope_scaling_in_cfg(cfg, cfg)
    if not patched_paths:
        return model_name

    hash_key = hashlib.md5(
        f"{model_path.resolve()}::{json.dumps(cfg, sort_keys=True)}".encode("utf-8")
    ).hexdigest()[:12]
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
        print(
            f"[llm_utils] Patched rope_scaling for vLLM compatibility at {', '.join(patched_paths)}: {patched_cfg_path}",
            flush=True,
        )
    else:
        print(
            f"[llm_utils] Using existing patched model config for {', '.join(patched_paths)}: {patched_cfg_path}",
            flush=True,
        )

    return str(patched_dir)


def _build_openai_client(llm_backend: str, api_base: str = None, api_key_env: str = "OPENAI_API_KEY"):
    client_kwargs = {}
    if llm_backend == "openai_compatible":
        if not api_base:
            raise ValueError("api_base is required when llm_backend=openai_compatible")
        client_kwargs["base_url"] = api_base

    api_key = os.environ.get(api_key_env)
    if api_key:
        client_kwargs["api_key"] = api_key
    elif llm_backend == "openai_compatible":
        client_kwargs["api_key"] = "EMPTY"

    return OpenAI(**client_kwargs)


def llm_init(
    model_name,
    tensor_parallel_size=1,
    max_seq_len_to_capture=8192,
    max_tokens=4000,
    seed=0,
    temperature=0,
    frequency_penalty=0,
    llm_backend="auto",
    api_base=None,
    api_key_env="OPENAI_API_KEY",
    request_model_name=None,
):
    resolved_backend = normalize_llm_backend(llm_backend, model_name)
    request_model_name = request_model_name or model_name

    if resolved_backend == "local_vllm":
        if LLM is None or SamplingParams is None:
            raise RuntimeError(
                "vLLM is not available in the current Python environment, but llm_backend=local_vllm was requested. "
                "Install vllm or switch to llm_backend=openai_compatible."
            ) from _VLLM_IMPORT_ERROR

        model_for_vllm = _prepare_local_model_path_for_vllm(model_name)
        client = LLM(
            model=model_for_vllm,
            tensor_parallel_size=tensor_parallel_size,
            max_seq_len_to_capture=max_seq_len_to_capture,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
        )
        llm = partial(client.chat, sampling_params=sampling_params, use_tqdm=False)
        return llm, resolved_backend

    client = _build_openai_client(resolved_backend, api_base=api_base, api_key_env=api_key_env)
    llm = partial(
        client.chat.completions.create,
        model=request_model_name,
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return llm, resolved_backend


def get_outputs(outputs, llm_backend):
    if llm_backend == "local_vllm":
        return outputs[0].outputs[0].text
    return outputs.choices[0].message.content


def llm_inf(llm, prompts, mode, llm_backend):
    res = []
    if "sys" in mode:
        conversation = [{"role": "system", "content": prompts["sys_query"]}]

    if "icl" in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    if "sys" in mode:
        conversation.append({"role": "user", "content": prompts["user_query"]})
        outputs = get_outputs(llm(messages=conversation), llm_backend)
        res.append(outputs)

    if "sys_cot" in mode:
        if "clear" in mode:
            conversation = []
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts["cot_query"]})
        outputs = get_outputs(llm(messages=conversation), llm_backend)
        res.append(outputs)
    elif "dc" in mode:
        if "ans:" not in res[0].lower() or "ans: not available" in res[0].lower() or "ans: no information available" in res[0].lower():
            conversation.append({"role": "user", "content": prompts["cot_query"]})
            outputs = get_outputs(llm(messages=conversation), llm_backend)
            res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res


def llm_inf_with_retry(llm, each_qa, llm_mode, llm_backend, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            return llm_inf(llm, each_qa, llm_mode, llm_backend)
        except openai.RateLimitError:
            wait_time = (2 ** retries) * 5
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
        except openai.APIConnectionError as e:
            if llm_backend == "openai_compatible":
                raise RuntimeError(
                    "Failed to reach the configured OpenAI-compatible endpoint. "
                    "Check that the local vLLM server is running and api_base points to /v1."
                ) from e
            raise
        except openai.NotFoundError as e:
            if llm_backend == "openai_compatible":
                raise RuntimeError(
                    "The OpenAI-compatible server did not recognize the requested model. "
                    "Check request_model_name against the server's --served-model-name."
                ) from e
            raise
    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")


def llm_inf_all(llm, each_qa, llm_mode, llm_backend, max_retries=5):
    if llm_backend == "local_vllm":
        return llm_inf(llm, each_qa, llm_mode, llm_backend)
    return llm_inf_with_retry(llm, each_qa, llm_mode, llm_backend, max_retries)
