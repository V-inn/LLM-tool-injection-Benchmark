"""results.py — benchmark results aggregation API (stub)"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("")
async def get_results():
    return {
        "has_results": False,
        "summary": {},
        "matrix": {},
        "awareness_cats": [],
        "lever_cats": [],
        "defenses": [],
    }


@router.get("/grades")
async def get_grades():
    return {"grades": {}}


@router.get("/validity")
async def get_validity():
    return {"validity": {}, "ref_model": "—", "threshold": 0.1}


@router.get("/delta")
async def get_delta():
    return {"delta": {}, "ref_model": "—"}


@router.get("/download")
async def download_results():
    return JSONResponse({"error": "No results"}, status_code=404)


@router.post("/clear")
async def clear_results():
    return {"ok": True}
