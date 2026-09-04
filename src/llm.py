"""
Provider-agnostic LLM call. Every node imports call_llm() from here rather
than talking to OpenAI/Ollama directly, so swapping providers touches one
function, not every node -- same pattern as project 1's generate.py.
"""

from __future__ import annotations

import json

from src import config


def call_llm(prompt: str, temperature: float = 0.1, json_mode: bool = False) -> str:
    if config.LLM_PROVIDER == "ollama":
        return _call_ollama(prompt, temperature, json_mode)
    elif config.LLM_PROVIDER == "openai":
        return _call_openai(prompt, temperature, json_mode)
    raise ValueError(f"Unknown LLM_PROVIDER: {config.LLM_PROVIDER}")


def _call_ollama(prompt: str, temperature: float, json_mode: bool) -> str:
    import requests

    payload = {
        "model": config.OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    resp = requests.post(f"{config.OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


def _call_openai(prompt: str, temperature: float, json_mode: bool) -> str:
    from openai import OpenAI

    client = OpenAI()
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(
        model=config.LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        **kwargs,
    )
    return resp.choices[0].message.content


def parse_json_response(raw: str) -> dict:
    """Best-effort JSON parse -- strips markdown code fences if present."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())
