"""
json_utils.py — Helpers for parsing JSON out of LLM responses.

Instruction-tuned models frequently wrap their JSON in markdown code fences
(```json ... ```) even when explicitly told not to, and even when the API is asked
to constrain output to JSON. Before packaging, the same fence-stripping regex pair was
copy-pasted in four places (both generators, the injection replacer, and the judge).
It lives here once now.
"""
from __future__ import annotations

import json
import re

# Matches an opening ```json / ``` fence (with optional language + whitespace) and a
# trailing ``` fence. Applied separately so a response with only one of them is still
# cleaned.
_FENCE_OPEN = re.compile(r'^```(?:json)?\s*')
_FENCE_CLOSE = re.compile(r'\s*```$')


def strip_code_fences(text: str) -> str:
    """Removes a leading/trailing markdown code fence from ``text`` and trims it."""
    text = _FENCE_OPEN.sub('', text.strip())
    text = _FENCE_CLOSE.sub('', text).strip()
    return text


def parse_json_response(text: str) -> dict:
    """
    Strips any markdown fences from an LLM response and parses it as JSON.

    Raises json.JSONDecodeError if the cleaned text is not valid JSON — callers
    decide whether to retry, dump the raw output, or propagate.
    """
    return json.loads(strip_code_fences(text))
