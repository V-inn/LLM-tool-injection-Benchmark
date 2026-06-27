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

from rbac_benchmark.ui.data import get_available_local_models, ollama_is_online

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


def _ctx(request: Request, page: str) -> dict:
    return {"request": request, "active_page": page, "pages": _PAGES}


@app.get("/")
def root():
    return RedirectResponse("/control")


@app.get("/control")
def page_control(request: Request):
    return templates.TemplateResponse("control.html", _ctx(request, "control"))


@app.get("/dashboard")
def page_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", _ctx(request, "dashboard"))


@app.get("/payload")
def page_payload(request: Request):
    return templates.TemplateResponse("payload.html", _ctx(request, "payload"))


@app.get("/prompts")
def page_prompts(request: Request):
    return templates.TemplateResponse("prompts.html", _ctx(request, "prompts"))


@app.get("/kappa")
def page_kappa(request: Request):
    return templates.TemplateResponse("kappa.html", _ctx(request, "kappa"))
