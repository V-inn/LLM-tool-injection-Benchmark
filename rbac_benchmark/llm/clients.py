"""
clients.py — Shared inference-backend wrappers (Gemini cloud + Ollama local).

Both prompt generators (injection / defense) previously duplicated the Gemini call,
its exponential-backoff retry loop, and the GEMINI_API_KEY check. That plumbing lives
here once. The Judge (evaluation/llm_judge.py) keeps its own bespoke ``chat`` call
(it needs ``format="json"`` + a pinned temperature) but shares json_utils.

These are thin wrappers, not an abstraction layer — callers still own prompt
construction, JSON parsing, and their user-facing messages.
"""
from __future__ import annotations

import asyncio
import os

from .json_utils import parse_json_response


def gemini_api_key() -> str:
    """Returns the stripped GEMINI_API_KEY from the environment, or '' if unset."""
    return os.environ.get("GEMINI_API_KEY", "").strip()


async def gemini_generate_json(
    model: str,
    prompt: str,
    system_instruction: str,
    max_retries: int = 3,
    base_delay: int = 2,
    retry_on_json_error: bool = False,
) -> str:
    """
    Calls the Google Gemini API and returns the raw response text (JSON-shaped).

    Requests ``response_mime_type="application/json"`` and reinforces the JSON-only
    rule via the system instruction, then retries transient API failures with
    exponential backoff (base_delay * 2**attempt). If ``retry_on_json_error`` is set,
    a response that fails to parse as JSON is also retried — the blue-team generator
    relies on this; the red-team generator parses downstream and handles bad JSON itself.

    The caller is responsible for validating the API key first (see gemini_api_key)
    so it can emit its own actionable message; genai.Client() reads the key from env.
    """
    from google import genai
    from google.genai import types

    client = genai.Client()
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
    )

    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model=model, contents=prompt, config=config
            )
            text = response.text
            if retry_on_json_error:
                # Validate parseability; a JSONDecodeError falls through to a retry.
                parse_json_response(text)
            return text
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"[-] Gemini API error: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)


async def ollama_generate(
    model: str,
    prompt: str,
    host: str = "http://127.0.0.1:11434",
    timeout: float = 120.0,
) -> str:
    """
    Routes a single-message generation to a local Ollama instance and returns the
    assistant's text. The 120s default timeout guards against slow local models.
    """
    from ollama import AsyncClient

    client = AsyncClient(host=host, timeout=timeout)
    response = await client.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
