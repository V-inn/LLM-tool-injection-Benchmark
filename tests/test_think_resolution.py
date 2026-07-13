"""
test_think_resolution.py — the explicit-only contract of the native-think channel.

The think mode sent to Ollama is derived from the model LABEL alone:

  * "<tag>#think"   -> (tag, True)
  * "<tag>#nothink" -> (tag, False)
  * bare tag        -> (tag, None)   # think omitted from the request entirely

There is deliberately NO family/prefix list that flips think on implicitly — a
bare tag must produce the exact pre-feature request for EVERY model, otherwise
the think/nothink ablation stops isolating the variable it exists to isolate.
"""
import inspect

from rbac_benchmark.orchestration.master_node import resolve_model_think


def test_explicit_suffixes_are_parsed():
    assert resolve_model_think("qwen3.5:29b#think") == ("qwen3.5:29b", True)
    assert resolve_model_think("qwen3.5:29b#nothink") == ("qwen3.5:29b", False)


def test_bare_tag_never_sets_think_for_any_family():
    # Including families with native thinking support: no name-based heuristic.
    for tag in ("qwen3.5:29b", "deepseek-r1:14b", "gemma4:9b", "ministral-3:8b"):
        assert resolve_model_think(tag) == (tag, None)


def test_unknown_suffix_falls_through_untouched():
    # A '#' that is not a think marker is part of the label, not a directive.
    assert resolve_model_think("weird#tag") == ("weird#tag", None)


def test_no_capability_list_resurfaces():
    # Guard against the config-driven prefix list coming back under another name:
    # the resolver must be a pure function of the label.
    params = list(inspect.signature(resolve_model_think).parameters)
    assert params == ["model"]
