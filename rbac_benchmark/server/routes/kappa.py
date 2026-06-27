"""kappa.py — κ validation + blind annotation API (stub)"""
from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def get_kappa():
    return {"kappa": None, "kappa_a": None, "kappa_b": None}


@router.get("/sample")
async def get_sample():
    return {"sample": [], "annotations": {}, "breakdown": {}}


@router.post("/build-sample")
async def build_sample(body: dict):
    return {"sample": [], "breakdown": {}}


@router.post("/annotate")
async def annotate(body: dict):
    return {"ok": True}


@router.post("/compute")
async def compute_kappa():
    return {"kappa": None, "kappa_a": None, "kappa_b": None}
