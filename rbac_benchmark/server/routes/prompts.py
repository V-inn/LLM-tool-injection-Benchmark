"""prompts.py — custom system-prompt CRUD (stub)"""
from fastapi import APIRouter

router = APIRouter()


@router.get("")
async def list_prompts():
    return {"prompts": []}


@router.post("/save")
async def save_prompt(body: dict):
    return {"ok": True}


@router.post("/toggle/{key}")
async def toggle_prompt(key: str):
    return {"ok": True}


@router.delete("/delete/{key}")
async def delete_prompt(key: str):
    return {"ok": True}
