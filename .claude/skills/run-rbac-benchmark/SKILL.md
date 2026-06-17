---
name: run-rbac-benchmark
description: Build, launch, screenshot, and drive the Tool-Calling RBAC Resilience Benchmark (the Streamlit "LLM Red Team Benchmark" GUI, the master orchestrator, the worker daemon, and the analyzer metrics modules). Use when asked to run, start, serve, build, test, smoke-test, or screenshot this benchmark / dashboard / Streamlit app, or to verify a change to the metrics (TPR/FPR/delta-TPR), the config, the worker, or the GUI.
---

# Run the Tool-Calling RBAC Resilience Benchmark

A distributed benchmark that stress-tests how local LLMs honour RBAC directives
under tool-output prompt injection. The code is an installable package,
`rbac_benchmark` (subpackages: `core`, `llm`, `evaluation`, `generation`,
`orchestration`, `worker`, `ui`). Runnable surfaces:

- **Internal metrics modules** (`rbac_benchmark/core/config.py`,
  `rbac_benchmark/evaluation/`) — TPR / FPR / delta-TPR / attack-strength /
  Cohen's κ. **Most PRs touch this layer.** Covered by `tests/` (pytest).
- **Streamlit GUI** (`rbac_benchmark/ui/app.py`, "LLM Red Team Benchmark") — thin
  shell + `ui/data.py`, `ui/charts.py`, `ui/state.py`, `ui/tabs/*.py`.
- **Worker daemon** (`rbac_benchmark/worker/node.py`) — UDP discovery responder.

Runtime data artifacts (results, generated prompts, kappa worksheet) live in
`data/` (resolved via `rbac_benchmark/paths.py`; override with `RBAC_DATA_DIR`).

Everything here is driven by **`.claude/skills/run-rbac-benchmark/driver.py`**.
Paths below are relative to the **repo root**.

> A real benchmark run needs Ollama + pulled models, which this box does not
> have. The driver deliberately drives every surface **without** inference: the
> metrics modules run on a synthetic fixture, the GUI degrades gracefully when
> Ollama is offline, and the worker handshake is pure UDP. Actual inference is
> out of scope for this harness.

> **Platform note:** the README says "Linux only." That is the author's
> machine, not a hard requirement. Everything in this skill was verified on
> **Windows 11** with the repo's `.venv`. The only Linux-specific bits are the
> `start_*.sh` convenience scripts (they call `pkill`/`ollama serve`); the
> Python surfaces are cross-platform.

## Prerequisites

A Python 3.11 virtualenv already exists at `.venv/`. Use its interpreter
directly — **`.venv/Scripts/python.exe`** (Windows) — for every command.

Install the package editable (registers the `rbac-*` entry points and pulls deps),
plus `pytest` (smoke) and `selenium` (GUI screenshots only):

```bash
.venv/Scripts/python.exe -m pip install -e . pytest selenium
```

For `shot` (GUI screenshots) you also need **Firefox**. It was found at
`C:\Program Files\Mozilla Firefox\firefox.exe`; override with the `FIREFOX_BIN`
env var if elsewhere. Selenium 4 auto-downloads `geckodriver` on first use
(needs network once).

## Run (agent path)

One driver, four subcommands. Run from the repo root.

### 1. Metrics smoke — the layer most PRs touch (no GUI, no network)

```bash
.venv/Scripts/python.exe .claude/skills/run-rbac-benchmark/driver.py smoke
```

Runs `pytest -q` (the `tests/` suite: `InferenceMetrics` accounting, TPR/FPR,
delta-TPR, attack-strength, Cohen's κ known-value + degenerate cases, and the
Phase-3 stored-label extraction). Ends with `SMOKE OK`. This is the fastest way to
verify a change to the metrics math. (`.venv/Scripts/python.exe -m pytest -q` works
directly too.)

### 2. Worker daemon smoke (no Ollama)

```bash
.venv/Scripts/python.exe .claude/skills/run-rbac-benchmark/driver.py worker-smoke
```

Launches the worker daemon (`rbac_benchmark/worker/node.py`), sends `OLLAMA_MASTER_SEEKING` to UDP 5005,
asserts the `OLLAMA_READY` reply, tears the daemon down. Ends with
`WORKER SMOKE OK`.

### 3. GUI screenshot — launch Streamlit + drive Firefox, then tear down

```bash
# Control Center tab (default landing page)
.venv/Scripts/python.exe .claude/skills/run-rbac-benchmark/driver.py shot \
  --out .claude/skills/run-rbac-benchmark/screenshots/control_center.png

# Dashboard & Results tab (renders the seeded metrics + Plotly charts)
.venv/Scripts/python.exe .claude/skills/run-rbac-benchmark/driver.py shot \
  --tab "Dashboard" \
  --out .claude/skills/run-rbac-benchmark/screenshots/dashboard.png
```

`shot` seeds `data/benchmark_results.json` + `data/kappa_samples.json` with the
synthetic fixtures, starts Streamlit headless (on `rbac_benchmark/ui/app.py`) on
port 8501, waits for `/healthz`, waits for the SPA to **hydrate** (the title text
appears only after JS runs — a plain one-shot screenshot captures a blank shell),
optionally clicks a tab, screenshots the full page, then kills Streamlit and
Firefox. Ends with `SHOT OK -> <path>`. Screenshots land in
`.claude/skills/run-rbac-benchmark/screenshots/` (git-ignored). Pass `--no-seed`
to screenshot whatever data is already there. Tabs: "Control Center", "Payload
Generation", "Custom Prompts", "Dashboard", "Judge Validation".

### 4. Seed dashboard data only

```bash
.venv/Scripts/python.exe .claude/skills/run-rbac-benchmark/driver.py seed
```

Writes `data/benchmark_results.json` + `data/kappa_samples.json` (git-ignored) so
the dashboard + Judge Validation tab have data without a real run. `shot` does this.

## Run (human path)

`bash start_master.sh` (Linux) starts Ollama + the `rbac-dashboard` entry point and
opens a browser. Useless headless and assumes Ollama is installed. Cross-platform
equivalent without Ollama:

```bash
.venv/Scripts/python.exe -m streamlit run rbac_benchmark/ui/app.py
```

Then open http://localhost:8501. The sidebar will show a red "Failed to connect
to Ollama" error — expected without Ollama; the UI still works for viewing
results. Ctrl-C to stop.

## Direct invocation

The analyzer and kappa modules are standalone CLIs (entry points after
`pip install -e .`; the `python -m` form works without install too):

```bash
rbac-analyze data/benchmark_results.json                    # aggregate TPR/FPR report
rbac-analyze data/benchmark_results.json --delta-tpr        # marginal defense gain
rbac-analyze data/benchmark_results.json --validate-attacks # Phase 2 attack strength
rbac-kappa   data/kappa_samples.json                        # Phase 3 Cohen's κ (offline)
# equivalently: .venv/Scripts/python.exe -m rbac_benchmark.evaluation.analyzer ...
```

Results JSON shape: keys are `"Model | Defense | Attack"`, values are
`core.config.InferenceMetrics` field dicts. `data/benchmark_results.json` is
git-ignored; use `driver.py seed` to (re)create one.

## Test

`tests/` is a pytest suite covering the metric / extraction / κ math. Run it with
`.venv/Scripts/python.exe -m pytest -q` (or `driver.py smoke`, which wraps it). It
is the recommended sanity check after editing any `core`/`evaluation` module.

## Gotchas

- **Streamlit is a websocket SPA.** Firefox's one-shot `firefox --screenshot URL`
  captures a **blank dark shell** because the page hasn't hydrated yet. You must
  drive it with Selenium and wait for an element (the title) to appear — which
  is exactly what `driver.py shot` does. Don't "simplify" it back to a one-shot.
- **Use `save_full_page_screenshot`**, not `save_screenshot` — the dashboard is
  taller than the viewport and a viewport shot crops the charts.
- **Ollama offline is fine.** `get_available_local_models()` falls back to a
  hardcoded model list, so the GUI launches and renders without Ollama. A real
  benchmark *run* (clicking "Run Benchmark") still needs Ollama + models and is
  not exercised here.
- **`pyyaml` missing from the venv** despite being a dependency — imports of
  `core.config.from_yaml` (and anything importing yaml) fail until you
  `pip install pyyaml` (or re-run `pip install -e .`).
- **Modules use absolute package imports** (`from rbac_benchmark.core.config import …`),
  so the package must be importable: `pip install -e .`, or run from the repo root
  (which puts `rbac_benchmark` on `sys.path`). No more `cd master`.
- **`rbac-analyze` argparse uses `--baseline-defense`**, not `--baseline`
  (prefix-matching lets `--baseline` work too, but the real flag is the long one).
- **`start_master.sh` / `start_worker.sh` are Linux-only** (`pkill`, `ollama serve`,
  `xdg-open` for the cat GIF). Don't run them on Windows; launch Streamlit
  directly per the human-path command above.

## Troubleshooting

- `ModuleNotFoundError: No module named 'rbac_benchmark'` → `.venv/Scripts/python.exe -m pip install -e .` (or run from the repo root).
- `ModuleNotFoundError: No module named 'yaml'` → `.venv/Scripts/python.exe -m pip install pyyaml`.
- `ModuleNotFoundError: No module named 'selenium'` → `... pip install selenium` (only needed for `shot`).
- `shot` produces a blank/dark image → you bypassed the hydration wait; use
  `driver.py shot`, not a raw `firefox --screenshot`.
- Firefox not found → set `FIREFOX_BIN` to your `firefox.exe` path.
- `Streamlit did not become ready` → port 8501 is busy; pass `--port <n>`.
