from fastapi import APIRouter, HTTPException
from core.schemas import ChatMessage, ChatResponse
from core.data_store import data_store
from agents.query_agent import QueryAgent
from agents.smart_chat_agent import SmartChatAgent
from agents.insight_agent import InsightAgent

router = APIRouter(prefix="/chat", tags=["chat"])
query_agent = QueryAgent()
smart_chat_agent = SmartChatAgent()
insight_agent = InsightAgent()

@router.post("/", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Process a chat message using You.com Smart API with analytical reasoning"""
    try:
        # Get latest data session
        run_id = message.run_id or data_store.get_latest_run_id()
        if not run_id:
            return ChatResponse(
                answer="I don't have any data to analyze yet. Please upload a CSV file first.",
                visualizations=None
            )

        df = data_store.get_data(run_id)
        if df is None:
            return ChatResponse(
                answer="I couldn’t find your dataset. Try re-uploading it.",
                visualizations=None
            )

        # 💬 Use Smart Chat Agent with rich context (it builds its own context from df)
        result = await smart_chat_agent.process_query(
            question=message.question,
            df=df
        )

        # 🧠 Step 3: Return Smart Agent's reasoning
        return ChatResponse(
            answer=result.get("answer", "I couldn’t find any clear pattern."),
            visualizations=result.get("visualizations")
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chat processing error: {str(e)}")