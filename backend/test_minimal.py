"""
Minimal test server to debug CORS and import issues
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="InsightForge AI - Test")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Test server running"}

@app.get("/test-cors")
async def test_cors():
    return {"cors": "working", "message": "CORS is properly configured"}

@app.post("/analyze/")
async def test_analyze(file: UploadFile = File(...)):
    """Minimal analyze endpoint for testing"""
    try:
        print(f"📁 Received file: {file.filename}")
        contents = await file.read()
        print(f"📊 File size: {len(contents)} bytes")
        
        # Just return a simple response without ETL processing
        return {
            "run_id": "test-123",
            "kpis": [{"name": "Test KPI", "value": 100}],
            "charts": [{"type": "test", "spec": {}}],
            "keywords": ["test"],
            "insights": ["Test insight"],
            "summary": "Test analysis completed successfully"
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
