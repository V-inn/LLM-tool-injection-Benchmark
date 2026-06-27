"""payloads.py — injection/defense generation API (stub)"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/injections")
async def list_injections():
    return {"injections": [], "generated_at": None}


@router.post("/generate-injections")
async def generate_injections(body: dict):
    return {"injections": [], "count": 0}


@router.post("/generate-defenses")
async def generate_defenses(body: dict):
    return {"defenses": [], "count": 0}


@router.get("/calibration")
async def get_calibration():
    return {"validity": {}}
