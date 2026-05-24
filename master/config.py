"""
config.py — Shared types, dataclasses, and configuration.

Single source of truth for:
  - Outcome enum (all possible inference results)
  - InferenceMetrics dataclass (typed result accumulator)
  - BenchmarkConfig dataclass (all run parameters)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------

class Outcome(str, Enum):
    """All possible outcomes of a single inference attempt."""
    COMPLIANT          = "compliant"
    SEVERITY_1         = "severity_1_violation"
    SEVERITY_2         = "severity_2_violation"
    SEVERITY_3         = "severity_3_violation"
    CONFUSION          = "confusion"
    NO_TOOL_CALLED     = "failure_no_tool_called"


# ---------------------------------------------------------------------------
# InferenceMetrics
# ---------------------------------------------------------------------------

@dataclass
class InferenceMetrics:
    """
    Typed accumulator for one (model, defense, attack) permutation.
    Replaces the untyped dict — typos become AttributeErrors, not silent bugs.
    """
    compliant:              int = 0
    severity_1_violation:   int = 0
    severity_2_violation:   int = 0
    severity_3_violation:   int = 0
    confusion:              int = 0
    failure_no_tool_called: int = 0
    coerced_violations:     int = 0
    raw_texts:              list[str] = field(default_factory=list)
    judge_reasoning:        list[str] = field(default_factory=list)

    def record(self, outcome: Outcome) -> None:
        """Increment the counter that corresponds to this outcome."""
        attr = outcome.value          # e.g. "compliant", "severity_3_violation"
        if hasattr(self, attr):
            setattr(self, attr, getattr(self, attr) + 1)

    @property
    def total_inferences(self) -> int:
        return (
            self.compliant
            + self.severity_1_violation
            + self.severity_2_violation
            + self.severity_3_violation
            + self.confusion
            + self.failure_no_tool_called
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> InferenceMetrics:
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# BenchmarkConfig
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """All parameters that control a benchmark run."""

    # Inference
    models:               list[str] = field(default_factory=lambda: ["ministral-3:8b", "qwen3.5:9b", "gemma4:e4b"])
    iterations:           int   = 5
    concurrency_per_node: int   = 2
    max_retries:          int   = 3

    # Discovery
    timeout:              float = 5.0
    exclude_master:       bool  = False
    workers_file:         str   = "workers.json"

    # Judge
    use_judge:            bool  = False
    judge_model:          str   = "qwen3.5:9b"

    # Observability
    show_thoughts: bool  = False

    # I/O
    output:               str | None = None
    checkpoint_path:      str   = "checkpoint.json"

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_json(cls, path: str) -> BenchmarkConfig:
        """Load config from a JSON file (useful for reproducible runs)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str) -> BenchmarkConfig:
        """Load config from a YAML file. Requires PyYAML."""
        try:
            import yaml
        except ImportError:
            raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, path: str) -> None:
        """Persist this config to disk for reproducibility."""
        Path(path).write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )