import time
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

try:
    from ollama import Client as OllamaClient
    from ollama import ResponseError as OllamaResponseError
    _OLLAMA_IMPORT_ERROR = None
except Exception as e:
    OllamaClient = None
    OllamaResponseError = Exception
    _OLLAMA_IMPORT_ERROR = e

from prompts import icl_user_prompt, icl_ass_prompt


VALID_LLM_BACKENDS = {"auto", "local_vllm", "openai", "ollama"}


def normalize_llm_backend(llm_backend: str, model_name: str) -> str:
    backend = (llm_backend or "auto").strip().lower()
    if backend not in VALID_LLM_BACKENDS:
        raise ValueError(f"Unsupported llm backend: {llm_backend}. Expected one of {sorted(VALID_LLM_BACKENDS)}")
    if backend == "auto":
        return "openai" if "gpt" in model_name else "local_vllm"
    return backend


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
):
    resolved_backend = normalize_llm_backend(llm_backend, model_name)

    if resolved_backend == "local_vllm":
        if LLM is None or SamplingParams is None:
            raise RuntimeError(
                "vLLM is not available in the current Python environment, but llm_backend=local_vllm was requested. "
                "Install vllm or switch to llm_backend=ollama."
            ) from _VLLM_IMPORT_ERROR
        client = LLM(
            model=model_name,
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
        return llm, resolved_backend

    client = OpenAI()
    llm = partial(
        client.chat.completions.create,
        model=model_name,
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return llm, resolved_backend


def get_outputs(outputs, llm_backend):
    if llm_backend == "local_vllm":
        return outputs[0].outputs[0].text
    if llm_backend == "ollama":
        return outputs.message.content
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
