from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io
from routers.analyze import generate_analysis
from core.data_store import data_store
import uuid

router = APIRouter(prefix="/analyze", tags=["analyze"])

@router.post("/")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        print(f"📁 Received file: {file.filename}, size: {file.size}")
        contents = await file.read()
        print(f"📊 File read successfully, {len(contents)} bytes")
        
        # Parse CSV
        df = pd.read_csv(io.BytesIO(contents))
        print(f"✅ CSV parsed: {len(df)} rows, {len(df.columns)} columns")
        
        # Use your new analysis logic
        analysis = generate_analysis(df)
        print("✅ Analysis completed successfully")
        
        # Create response in expected format
        run_id = str(uuid.uuid4())
        
        # Store data for chat
        data_store.store_data(run_id, df, {
            'kpis': analysis['kpis'],
            'charts': analysis['charts'],
            'trends': analysis['trends'],
            'deltas': analysis['deltas'],
            'correlations': analysis['correlations'],
            'summary': analysis['summary'],
            'qa_context': analysis['qa_context']
        })
        
        # Return in format expected by frontend
        return {
            "run_id": run_id,
            "kpis": [{"name": kpi["metric"], "value": float(kpi["mean"]), "trend": analysis['deltas'].get(kpi['metric'], {}).get('direction', 'stable')} for kpi in analysis["kpis"]],
            "charts": analysis["charts"],
            "keywords": df.columns.tolist()[:5],
            "insights": analysis["summary"]["highlights"],
            "summary": analysis["summary"]["overall_assessment"],
            "deltas": analysis["deltas"],
            "correlations": analysis["correlations"]["strong_pairs"]
        }
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
