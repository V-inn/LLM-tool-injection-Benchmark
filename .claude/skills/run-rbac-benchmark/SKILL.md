---
name: run-rbac-benchmark
description: Build, launch, screenshot, and drive the Tool-Calling RBAC Resilience Benchmark (FastAPI "LLM Red Team Benchmark" web UI, the master orchestrator, the worker daemon, and the analyzer metrics modules). Use when asked to run, start, serve, build, test, smoke-test, or screenshot this benchmark / dashboard / FastAPI app, or to verify a change to the metrics (TPR/FPR/delta-TPR), the config, the worker, or the UI.
---

# Run the Tool-Calling RBAC Resilience Benchmark

A distributed benchmark that stress-tests how local LLMs honour RBAC directives
under tool-output prompt injection. The code is an installable package,
`rbac_benchmark` (subpackages: `core`, `llm`, `evaluation`, `generation`,
`orchestration`, `worker`, `server`). Runnable surfaces:

- **Internal metrics modules** (`rbac_benchmark/core/config.py`,
  `rbac_benchmark/evaluation/`) — TPR / FPR / delta-TPR / attack-strength /
  Cohen's κ. **Most PRs touch this layer.** Covered by `tests/` (pytest).
- **FastAPI web server** (`rbac_benchmark/server/app.py`) — Jinja2-rendered HTML
  pages at `/control`, `/dashboard`, `/payload`, `/prompts`, `/kappa`; all data
  loaded client-side via `/api/*` JSON endpoints. Launch with uvicorn on port 8000.
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

### 3. GUI screenshot — launch FastAPI/uvicorn + drive Firefox, then tear down

```bash
# Control Center (default landing page)
.venv/Scripts/python.exe .claude/skills/run-rbac-benchmark/driver.py shot \
  --out .claude/skills/run-rbac-benchmark/screenshots/control.png

# Dashboard page (renders the seeded metrics + Chart.js charts)
.venv/Scripts/python.exe .claude/skills/run-rbac-benchmark/driver.py shot \
  --page dashboard \
  --out .claude/skills/run-rbac-benchmark/screenshots/dashboard.png
```

`shot` seeds `data/benchmark_results.json` + `data/kappa_samples.json` with the
synthetic fixtures, starts uvicorn headless on port 8000, waits for `/api/docs` to
return 200, navigates Firefox to the target page, waits for the page title to
appear, screenshots the full page, then tears everything down. Ends with
`SHOT OK -> <path>`. Screenshots land in
`.claude/skills/run-rbac-benchmark/screenshots/` (git-ignored). Pass `--no-seed`
to screenshot whatever data is already there.

Pages (pass to `--page`): `control`, `dashboard`, `payload`, `prompts`, `kappa`.

### 4. Seed dashboard data only

```bash
.venv/Scripts/python.exe .claude/skills/run-rbac-benchmark/driver.py seed
```

Writes `data/benchmark_results.json` + `data/kappa_samples.json` (git-ignored) so
the dashboard + Judge κ page have data without a real run. `shot` does this automatically.

## Run (human path)

`bash start_master.sh` (Linux) starts Ollama and opens a browser. Useless headless.
Cross-platform equivalent without Ollama:

```bash
.venv/Scripts/python.exe -m uvicorn rbac_benchmark.server.app:app --reload --port 8000
```

Then open http://localhost:8000. The nav rail Ollama status shows "checking…" /
offline — expected without Ollama; all pages still render. Ctrl-C to stop.

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

- **FastAPI nav uses `<a>` links, not tab buttons.** `base.html` renders
  `<a href="/page_id" class="nav-item">` elements in a sidebar nav rail — NOT
  `<button role="tab">`. `driver.py shot` navigates to the target page by URL
  (`--page <route>`). Don't write Selenium XPaths for `button[@role='tab']`.
- **`/healthz` does not exist.** The readiness probe in `driver.py` polls
  `/api/docs` (FastAPI auto-serves this at 200). If you add a custom `/healthz`,
  update `_wait_http` in driver.py.
- **Use `save_full_page_screenshot`**, not `save_screenshot` — the dashboard is
  taller than the viewport and a viewport shot crops the charts.
- **Ollama offline is fine.** `get_available_local_models()` returns `[]`; the
  nav cluster shows offline. A real benchmark *run* (POST `/api/run/start`) still
  needs Ollama + models and is not exercised here.
- **`pyyaml` missing from the venv** despite being a dependency — imports of
  `core.config.from_yaml` (and anything importing yaml) fail until you
  `pip install pyyaml` (or re-run `pip install -e .`).
- **Modules use absolute package imports** (`from rbac_benchmark.core.config import …`),
  so the package must be importable: `pip install -e .`, or run from the repo root
  (which puts `rbac_benchmark` on `sys.path`). No more `cd master`.
- **`rbac-analyze` argparse uses `--baseline-defense`**, not `--baseline`
  (prefix-matching lets `--baseline` work too, but the real flag is the long one).
- **`start_master.sh` / `start_worker.sh` are Linux-only** (`pkill`, `ollama serve`,
  `xdg-open` for the cat GIF). Don't run them on Windows; launch uvicorn directly
  per the human-path command above.

## Troubleshooting

- `ModuleNotFoundError: No module named 'rbac_benchmark'` → `.venv/Scripts/python.exe -m pip install -e .` (or run from the repo root).
- `ModuleNotFoundError: No module named 'yaml'` → `.venv/Scripts/python.exe -m pip install pyyaml`.
- `ModuleNotFoundError: No module named 'selenium'` → `... pip install selenium` (only needed for `shot`).
- `shot` produces a blank/dark image → you bypassed the hydration wait; use
  `driver.py shot`, not a raw `firefox --screenshot`.
- Firefox not found → set `FIREFOX_BIN` to your `firefox.exe` path.
- `uvicorn did not become ready` → port 8000 is busy; pass `--port <n>`.
