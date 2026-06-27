"""
server/app.py — FastAPI entry point.

Launch with:
    uvicorn rbac_benchmark.server.app:app --reload --port 8000

Page routes render Jinja2 templates (static HTML shells).
All data is loaded client-side via /api/* JSON endpoints.
"""
from __future__ import annotations

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
    return {"models": get_available_local_models(), "online": ollama_is_online()}

app.include_router(_sys, prefix="/api/system", tags=["system"])


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
