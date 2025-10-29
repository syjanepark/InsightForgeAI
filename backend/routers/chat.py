# chat.py
from fastapi import APIRouter, HTTPException
from core.schemas import ChatMessage, ChatResponse
from core.data_store import data_store
from agents.smart_chat_agent import SmartChatAgent

router = APIRouter(prefix="/chat", tags=["chat"])
smart = SmartChatAgent()

@router.post("/", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        print(f"💬 Chat request: {message.question}")
        
        run_id = message.run_id or data_store.get_latest_run_id()
        if not run_id:
            return ChatResponse(
                answer="Upload a CSV first so I can analyze it.", 
                visualizations=None,
                suggested_actions=None,
                citations=None
            )

        analysis = data_store.get_metadata(run_id)
        if not analysis:
            return ChatResponse(
                answer="I can't find the analysis for this session. Try re-uploading.", 
                visualizations=None,
                suggested_actions=None,
                citations=None
            )

        # Get the dataframe for more detailed analysis
        df = data_store.get_data(run_id)
        context = analysis.get("qa_context", "No context available")

        print(f"📊 Processing question with context: {len(context)} chars")

        # let the agent answer with grounded reasoning and access to raw data
        result = await smart.process_query(question=message.question, context=context, df=df)

        return ChatResponse(
            answer=result.get("answer", "I couldn't process your question."),
            visualizations=result.get("visualizations"),
            suggested_actions=result.get("suggested_actions"),
            citations=result.get("citations")
        )
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return a proper ChatResponse instead of raising HTTPException to avoid CORS issues
        return ChatResponse(
            answer="I encountered an error processing your question. Please try again.",
            visualizations=None,
            suggested_actions=None,
            citations=None
        )