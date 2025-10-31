from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze_endpoint, chat, chart_preview
from dotenv import load_dotenv

# Ensure environment variables from .env are loaded at startup
load_dotenv()

app = FastAPI(title="InsightForge AI")

# CORS - Specific configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",  # Alternative localhost
        "https://localhost:3000",  # HTTPS version
        "*"  # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Routers
app.include_router(analyze_endpoint.router)
app.include_router(chat.router)
app.include_router(chart_preview.router)

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "InsightForge AI Backend is running"}

@app.get("/test-cors")
async def test_cors():
    return {"cors": "working", "message": "CORS is properly configured"}