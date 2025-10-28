from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze, chat

app = FastAPI(title="InsightForge AI")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(analyze.router)
app.include_router(chat.router)

@app.get("/health")
async def health():
    return {"ok": True}