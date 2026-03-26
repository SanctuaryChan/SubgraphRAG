import inspect
import time
from functools import partial
from pathlib import Path
import re

import openai
from openai import OpenAI

try:
    from vllm import LLM, SamplingParams
    _VLLM_IMPORT_ERROR = None
except Exception as e:
    LLM = None
    SamplingParams = None
    _VLLM_IMPORT_ERROR = e

try:
    from vllm.engine.arg_utils import EngineArgs as VllmEngineArgs
except Exception:
    VllmEngineArgs = None

try:
    from ollama import Client as OllamaClient
    from ollama import ResponseError as OllamaResponseError
    _OLLAMA_IMPORT_ERROR = None
except Exception as e:
    OllamaClient = None
    OllamaResponseError = Exception
    _OLLAMA_IMPORT_ERROR = e

try:
    from transformers import AutoTokenizer as TransformersAutoTokenizer
    _TRANSFORMERS_IMPORT_ERROR = None
except Exception as e:
    TransformersAutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = e

try:
    from modelscope import AutoTokenizer as ModelScopeAutoTokenizer
    _MODELSCOPE_IMPORT_ERROR = None
except Exception as e:
    ModelScopeAutoTokenizer = None
    _MODELSCOPE_IMPORT_ERROR = e

from prompts import icl_user_prompt, icl_ass_prompt


VALID_LLM_BACKENDS = {"auto", "local_vllm", "openai", "ollama"}


def resolve_model_ref(model_name: str, local_model_path: str = None):
    if local_model_path:
        path = Path(local_model_path).expanduser()
        if path.exists():
            return str(path.resolve()), "local_path"

    path = Path(model_name).expanduser()
    if path.exists():
        return str(path.resolve()), "local_path"
    return model_name, "model_ref"


def is_qwen3_model(model_ref: str) -> bool:
    lowered = (model_ref or "").lower()
    return "qwen3" in lowered and "qwen3.5" not in lowered and "qwen35" not in lowered


def load_chat_tokenizer(model_ref: str):
    errors = []
    loaders = [
        ("transformers", TransformersAutoTokenizer, _TRANSFORMERS_IMPORT_ERROR),
        ("modelscope", ModelScopeAutoTokenizer, _MODELSCOPE_IMPORT_ERROR),
    ]

    for loader_name, loader, import_error in loaders:
        if loader is None:
            if import_error is not None:
                errors.append(f"{loader_name} import failed: {type(import_error).__name__}: {import_error}")
            continue
        try:
            return loader.from_pretrained(model_ref, trust_remote_code=True)
        except Exception as e:
            errors.append(f"{loader_name} load failed: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Unable to load a tokenizer for the requested Qwen3 model. "
        f"Attempted model_ref={model_ref!r}. Errors: {' | '.join(errors)}"
    )


def sanitize_model_output(output_text: str) -> str:
    if not output_text:
        return ""

    cleaned = output_text.strip()
    lowered = cleaned.lower()
    if "</think>" in lowered:
        match = list(re.finditer(r"(?i)</think>", cleaned))
        if match:
            cleaned = cleaned[match[-1].end():]
    elif "<think>" in lowered:
        ans_match = re.search(r"(?i)\bans\s*:", cleaned)
        if ans_match is not None:
            cleaned = cleaned[ans_match.start():]

    cleaned = re.sub(r"(?is)<think>.*?</think>", "", cleaned)
    cleaned = re.sub(r"(?i)</?think>", "", cleaned)
    return cleaned.strip()


def normalize_llm_backend(llm_backend: str, model_name: str) -> str:
    backend = (llm_backend or "auto").strip().lower()
    if backend not in VALID_LLM_BACKENDS:
        raise ValueError(f"Unsupported llm backend: {llm_backend}. Expected one of {sorted(VALID_LLM_BACKENDS)}")
    if backend == "auto":
        return "openai" if "gpt" in model_name else "local_vllm"
    return backend


def _supports_signature_kwarg(callable_obj, arg_name: str) -> bool:
    if callable_obj is None:
        return False
    try:
        return arg_name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


def get_vllm_init_kwargs(model_ref: str, tensor_parallel_size: int, max_seq_len_to_capture: int):
    kwargs = {
        "model": model_ref,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
    }
    runtime_info = {
        "requested_max_seq_len_to_capture": max_seq_len_to_capture,
        "applied_max_seq_len_to_capture": False,
    }

    if _supports_signature_kwarg(getattr(VllmEngineArgs, "__init__", None), "max_seq_len_to_capture"):
        kwargs["max_seq_len_to_capture"] = max_seq_len_to_capture
        runtime_info["applied_max_seq_len_to_capture"] = True

    return kwargs, runtime_info


def llm_init(
    model_name,
    tensor_parallel_size=1,
    max_seq_len_to_capture=8192,
    max_tokens=4000,
    seed=0,
    temperature=0,
    frequency_penalty=0,
    llm_backend="auto",
    ollama_host="http://127.0.0.1:11434",
    enable_thinking=False,
    local_model_path=None,
):
    resolved_backend = normalize_llm_backend(llm_backend, model_name)
    resolved_model_ref, resolved_model_source = resolve_model_ref(model_name, local_model_path)
    runtime_info = {
        "resolved_model_ref": resolved_model_ref,
        "resolved_model_source": resolved_model_source,
        "enable_thinking": bool(enable_thinking),
    }

    if resolved_backend == "local_vllm":
        if LLM is None or SamplingParams is None:
            raise RuntimeError(
                "vLLM is not available in the current Python environment, but llm_backend=local_vllm was requested. "
                "Install vllm or switch to llm_backend=ollama."
            ) from _VLLM_IMPORT_ERROR
        llm_init_kwargs, llm_init_runtime_info = get_vllm_init_kwargs(
            resolved_model_ref,
            tensor_parallel_size,
            max_seq_len_to_capture,
        )
        runtime_info.update(llm_init_runtime_info)
        if not runtime_info["applied_max_seq_len_to_capture"]:
            print(
                "Current vLLM build does not accept max_seq_len_to_capture; "
                "continuing without this optimization setting."
            )
        client = LLM(**llm_init_kwargs)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
        )
        if is_qwen3_model(resolved_model_ref):
            tokenizer = load_chat_tokenizer(resolved_model_ref)
            runtime_info["uses_qwen3_chat_template"] = True

            def llm(messages):
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=bool(enable_thinking),
                )
                outputs = client.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
                return outputs[0].outputs[0].text

            return llm, resolved_backend, runtime_info

        runtime_info["uses_qwen3_chat_template"] = False

        def llm(messages):
            outputs = client.chat(messages=messages, sampling_params=sampling_params, use_tqdm=False)
            return outputs[0].outputs[0].text

        return llm, resolved_backend, runtime_info

    if resolved_backend == "ollama":
        if OllamaClient is None:
            raise RuntimeError(
                "The ollama Python package is not installed, but llm_backend=ollama was requested. "
                "Install it with `pip install ollama`."
            ) from _OLLAMA_IMPORT_ERROR
        client = OllamaClient(host=ollama_host)
        options = {
            "temperature": temperature,
            "seed": seed,
            "num_predict": max_tokens,
            "num_ctx": max_seq_len_to_capture,
            "frequency_penalty": frequency_penalty,
        }
        llm = partial(client.chat, model=model_name, options=options)

        def invoke_ollama(messages):
            outputs = llm(messages=messages)
            return outputs.message.content

        runtime_info["uses_qwen3_chat_template"] = False
        return invoke_ollama, resolved_backend, runtime_info

    client = OpenAI()
    llm = partial(
        client.chat.completions.create,
        model=model_name,
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    def invoke_openai(messages):
        outputs = llm(messages=messages)
        message = outputs.choices[0].message
        return message.content or ""

    runtime_info["uses_qwen3_chat_template"] = False
    return invoke_openai, resolved_backend, runtime_info


def llm_inf(llm, prompts, mode, llm_backend):
    res = []
    raw_res = []
    conversation = []
    if "sys" in mode:
        conversation = [{"role": "system", "content": prompts["sys_query"]}]

    if "icl" in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    if "sys" in mode:
        conversation.append({"role": "user", "content": prompts["user_query"]})
        raw_output = llm(messages=conversation)
        outputs = sanitize_model_output(raw_output)
        res.append(outputs)
        raw_res.append(raw_output)

    if "sys_cot" in mode:
        if "clear" in mode:
            conversation = []
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts["cot_query"]})
        raw_output = llm(messages=conversation)
        outputs = sanitize_model_output(raw_output)
        res.append(outputs)
        raw_res.append(raw_output)
    elif "dc" in mode:
        if "ans:" not in res[0].lower() or "ans: not available" in res[0].lower() or "ans: no information available" in res[0].lower():
            conversation.append({"role": "user", "content": prompts["cot_query"]})
            raw_output = llm(messages=conversation)
            outputs = sanitize_model_output(raw_output)
            res[0] = outputs
            raw_res[0] = raw_output
        res.append("")
        raw_res.append("")
    else:
        res.append("")
        raw_res.append("")

    return {"responses": res, "raw_responses": raw_res}


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
        except OllamaResponseError as e:
            message = str(e)
            if "model" in message.lower() and "not found" in message.lower():
                raise RuntimeError(
                    "Ollama could not find the requested model. "
                    "Check that the model has been pulled and model_name matches your Ollama tag."
                ) from e
            raise RuntimeError(f"Ollama request failed: {message}") from e
        except Exception as e:
            if llm_backend == "ollama":
                raise RuntimeError(
                    "Failed to reach the Ollama service. Check that Ollama is running and ollama_host is correct."
                ) from e
            raise
    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")


def llm_inf_all(llm, each_qa, llm_mode, llm_backend, max_retries=5):
    if llm_backend == "local_vllm":
        return llm_inf(llm, each_qa, llm_mode, llm_backend)
    return llm_inf_with_retry(llm, each_qa, llm_mode, llm_backend, max_retries)
