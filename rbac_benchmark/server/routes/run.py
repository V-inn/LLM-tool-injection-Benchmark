"""run.py — benchmark dispatch + SSE stream + thought inspector (stub)"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/thoughts")
async def get_thoughts():
    return {"thoughts": []}


@router.post("/start")
async def start_run(body: dict):
    return {"run_id": "not-implemented", "total_inferences": 0}


@router.post("/abort/{run_id}")
async def abort_run(run_id: str):
    return {"ok": True}


@router.get("/stream/{run_id}")
async def stream_run(run_id: str):
    from fastapi.responses import StreamingResponse

    async def gen():
        yield "data: __DONE__\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
