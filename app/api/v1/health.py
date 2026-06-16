from fastapi import APIRouter, Request
router = APIRouter()
@router.get("/ready")
async def ready(request: Request):
    return {"status": "ok", "detector": request.app.state.detector.name}