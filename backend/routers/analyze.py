from fastapi import APIRouter, UploadFile, File, HTTPException
from core.orchestrator import Orchestrator

router = APIRouter(prefix="/analyze", tags=["analyze"])

@router.post("/")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        orchestrator = Orchestrator()
        result = await orchestrator.run(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))