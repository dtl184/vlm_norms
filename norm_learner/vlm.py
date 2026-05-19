"""
VLM backend implementations.

Three backends are provided:

1. MockVLM — returns a stub response; useful for unit tests or dry runs.
2. OpenAICompatibleVLM — calls any OpenAI-compatible REST endpoint using
   ``requests`` (no extra packages required).  Works with:
   - Qwen via DashScope (base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
   - Local Ollama (base_url="http://localhost:11434/v1")
   - OpenAI itself
3. LocalTransformersVLM — runs a local Qwen (or other causal-LM) model via
   ``transformers``.  Requires the model weights to be downloaded.

Usage example
-------------
    from norm_learner.vlm import OpenAICompatibleVLM
    vlm = OpenAICompatibleVLM(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        model="qwen-plus",
    )
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class VLMInterface(ABC):
    """Abstract VLM backend."""

    @abstractmethod
    def query(self, prompt: str) -> str:
        """Send *prompt* to the model and return the raw text response."""


# ---------------------------------------------------------------------------
# Mock backend (testing / no-VLM baseline)
# ---------------------------------------------------------------------------

class MockVLM(VLMInterface):
    """
    Returns a deterministic stub response.  Useful for testing the pipeline
    end-to-end without a real model.
    """

    def __init__(self, fixed_response: str | None = None) -> None:
        self._response = fixed_response or json.dumps([
            {
                "type": "unknown",
                "description": "Mock VLM: no real model attached.",
                "reasoning": "MockVLM returns this stub for testing purposes.",
            }
        ])

    def query(self, prompt: str) -> str:  # noqa: ARG002
        logger.debug("MockVLM.query called (returning stub).")
        return self._response


# ---------------------------------------------------------------------------
# OpenAI-compatible REST backend
# ---------------------------------------------------------------------------

class OpenAICompatibleVLM(VLMInterface):
    """
    Calls any OpenAI-compatible chat-completions endpoint via ``requests``.

    Parameters
    ----------
    base_url : str
        API root, e.g. "https://dashscope.aliyuncs.com/compatible-mode/v1"
        or "http://localhost:11434/v1".
    api_key : str
        Bearer token / API key (use a dummy string for keyless local servers).
    model : str
        Model identifier, e.g. "qwen-plus", "qwen2.5-72b-instruct",
        "llama3", etc.
    system_prompt : str | None
        Optional system message prepended to every request.
    temperature : float
        Sampling temperature (0.0 for greedy).
    max_tokens : int
        Maximum tokens in the response.
    timeout : int
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 120,
    ) -> None:
        import requests  # lazy import so the class definition never fails
        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt or (
            "You are a social norm analyst. Respond only with valid JSON."
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def query(self, prompt: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        response = self._requests.post(
            url, json=payload, headers=headers, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Local transformers backend
# ---------------------------------------------------------------------------

class LocalTransformersVLM(VLMInterface):
    """
    Runs a local Hugging Face causal-LM (e.g. Qwen/Qwen2.5-7B-Instruct)
    using the ``transformers`` library.

    The model and tokenizer are loaded lazily on the first ``query`` call.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path, e.g.
        "Qwen/Qwen2.5-7B-Instruct".
    device : str
        "auto", "cuda", "cpu", etc.
    torch_dtype : str
        "auto", "bfloat16", "float16", "float32".
    max_new_tokens : int
        Generation budget.
    temperature : float
        Sampling temperature (0.0 uses greedy decoding).
    system_prompt : str | None
        Optional system message.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        system_prompt: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            "You are a social norm analyst. Respond only with valid JSON."
        )
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.torch_dtype, "auto")

        logger.info("Loading model %s …", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("Model loaded.")

    def query(self, prompt: str) -> str:
        if self._model is None:
            self._load()

        import torch

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt")
        # Move inputs to the same device as the model
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        do_sample = self.temperature > 0.0
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
            )

        # Strip the prompt tokens from the output
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)
