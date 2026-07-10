"""
server/app.py — FastAPI entry point.

Launch with:
    uvicorn rbac_benchmark.server.app:app --reload --port 8000

Page routes render Jinja2 templates (static HTML shells).
All data is loaded client-side via /api/* JSON endpoints.
"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

def get_available_local_models():
    try:
        import ollama
        models_dict = ollama.list()
        raw = models_dict.get("models", []) if isinstance(models_dict, dict) else getattr(models_dict, "models", [])
        out = []
        for m in raw:
            try:
                out.append(m["model"])
            except (TypeError, KeyError):
                out.append(getattr(m, "model", str(m)))
        return out
    except Exception:
        return []


# Capabilities are one `ollama show` (a POST /api/show) per installed model, and the
# sidebar polls /api/system/models every 10s from every open tab — without a cache
# that's a steady burst of N POSTs hitting the master's Ollama. Keyed by the exact
# model set so a pull/delete (seen via the still-per-poll ollama.list()) refreshes
# immediately; the TTL only covers the rare same-tag re-pull that changes capabilities.
_CAPS_TTL_SECONDS = 300.0
_caps_cache: dict[str, object] = {"key": None, "at": 0.0, "caps": {}}


def get_model_capabilities(models: list[str]) -> dict[str, list[str] | None]:
    """Best-effort map of model -> Ollama capability strings, None = unknown.

    `ollama show` reports e.g. ["completion", "tools", "thinking"] for thinking
    models and ["embedding"] for embedding-only ones. Older client/server
    versions don't return the field, and a model can disappear between list()
    and show() — those map to None so the UI can treat the capability set as
    unknown instead of guessing in either direction.
    """
    key = tuple(sorted(models))
    now = time.monotonic()
    if _caps_cache["key"] == key and now - _caps_cache["at"] < _CAPS_TTL_SECONDS:
        return dict(_caps_cache["caps"])  # type: ignore[arg-type]

    try:
        import ollama
    except Exception:
        return {m: None for m in models}
    caps: dict[str, list[str] | None] = {}
    for m in models:
        try:
            info = ollama.show(m)
            c = info.get("capabilities") if isinstance(info, dict) else getattr(info, "capabilities", None)
            caps[m] = list(c) if c is not None else None
        except Exception:
            caps[m] = None
    # A map with unknowns may just be a transient show() failure; caching it would
    # pin "unknown" (and thus disable/enable the wrong models) for the whole TTL.
    if all(v is not None for v in caps.values()):
        _caps_cache.update(key=key, at=now, caps=dict(caps))
    return caps


def ollama_is_online() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False

_HERE = Path(__file__).parent

app = FastAPI(title="LLM Red Team Benchmark", docs_url="/api/docs")

# ── Static files + Jinja2 ─────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")
templates = Jinja2Templates(directory=str(_HERE / "templates"))

# ── Import routers (registered below after definitions) ────────────────────────
from rbac_benchmark.server.routes import run, results, prompts, payloads, kappa  # noqa: E402

app.include_router(run.router,      prefix="/api/run",      tags=["run"])
app.include_router(results.router,  prefix="/api/results",  tags=["results"])
app.include_router(prompts.router,  prefix="/api/prompts",  tags=["prompts"])
app.include_router(payloads.router, prefix="/api/payloads", tags=["payloads"])
app.include_router(kappa.router,    prefix="/api/kappa",    tags=["kappa"])


# ── System info endpoints ──────────────────────────────────────────────────────
from fastapi import APIRouter as _AR  # noqa: E402

_sys = _AR()

@_sys.get("/models")
def get_models():
    models = get_available_local_models()
    return {
        "models": models,
        "capabilities": get_model_capabilities(models),
        "online": ollama_is_online(),
    }


@_sys.get("/cluster")
def get_cluster():
    """Live worker-node view for the sidebar (IP + online/offline), tracked by the
    server's own UDP discovery poller — see server/cluster.py."""
    return {"nodes": cluster.snapshot()}

app.include_router(_sys, prefix="/api/system", tags=["system"])


# ── Background cluster monitor ──────────────────────────────────────────────────
from rbac_benchmark.server import cluster  # noqa: E402


@app.on_event("startup")
def _start_cluster_monitor():
    # Start the discovery poller when the ASGI app boots (not on bare import), so
    # route introspection / tests don't spawn a broadcasting thread.
    cluster.start()


# ── Page routes ────────────────────────────────────────────────────────────────
_PAGES = {
    "control":   "Control Center",
    "dashboard": "Dashboard",
    "payload":   "Payload Gen",
    "prompts":   "Custom Prompts",
    "kappa":     "Judge κ",
}


def _ctx(page: str) -> dict:
    return {"active_page": page, "pages": _PAGES}


@app.get("/")
def root():
    return RedirectResponse("/control")


@app.get("/control")
def page_control(request: Request):
    return templates.TemplateResponse(request, "control.html", _ctx("control"))


@app.get("/dashboard")
def page_dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", _ctx("dashboard"))


@app.get("/payload")
def page_payload(request: Request):
    return templates.TemplateResponse(request, "payload.html", _ctx("payload"))


@app.get("/prompts")
def page_prompts(request: Request):
    return templates.TemplateResponse(request, "prompts.html", _ctx("prompts"))


@app.get("/kappa")
def page_kappa(request: Request):
    return templates.TemplateResponse(request, "kappa.html", _ctx("kappa"))
