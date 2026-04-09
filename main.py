"""
Hospital Video Agent — Main FastAPI Application

=== WHAT IS THIS FILE? ===
This is the ENTRY POINT of our application — like the "front door" of our hospital.
FastAPI creates a web server that listens for requests and routes them to the right handler.

=== HOW DOES FASTAPI WORK? ===

Think of FastAPI as a waiter in a restaurant:
1. A customer (frontend/browser) sends a request (order)
2. FastAPI receives it and routes it to the right function (kitchen)
3. The function processes the request and returns a response (food)

FastAPI uses "decorators" (@app.get, @app.post) to define which function 
handles which URL:
    @app.get("/health")  → When someone visits http://localhost:8000/health
    @app.post("/chat")   → When someone sends a chat message

=== WHAT IS CORS? ===
CORS (Cross-Origin Resource Sharing) = security rule.
By default, a website at localhost:3000 (React) can't talk to localhost:8000 (FastAPI).
CORS middleware tells the browser "it's OK, let them communicate."
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import sys
import asyncio
import httpx

# Set up logging so we can see what's happening in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# ==================== Create the FastAPI App ====================
# 
# Think of this as creating the "restaurant" itself.
# FastAPI() creates the app, and we configure it with:
#   - title: shown in the auto-generated API docs
#   - description: explains what this API does
#   - version: semantic versioning
#
app = FastAPI(
    title="🏥 Hospital Video Agent API",
    description="AI-powered video receptionist for MedCare General Hospital",
    version="1.0.0"
)

# ==================== CORS Middleware ====================
# 
# WHY: Our React frontend (localhost:3000) needs to talk to this API (localhost:8000).
# Without CORS, the browser would block the request for security reasons.
#
# allow_origins=["*"] = allow ALL websites to access this API (fine for development)
# In production, you'd restrict this to your actual domain.
#
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (for development)
    allow_credentials=True,     # Allow cookies/auth headers
    allow_methods=["*"],        # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],        # Allow all headers
)

# ==================== Initialize the Backend Systems ====================
import base64
import tempfile
import os
from backend.rag_pipeline import HospitalRAGPipeline
from backend.agent import HospitalAgent
from backend.speech_to_text import SpeechToText
from backend.text_to_speech import TextToSpeech
from backend.config import settings
from backend.did_client import DIDClient

logger.info("=" * 60)
logger.info("🏥 Starting Hospital Video Agent Modules...")
logger.info("=" * 60)

# 1. RAG limits knowledge strictly to hospital data
rag_pipeline = HospitalRAGPipeline()

# 2. Agent gives the AI the power to use tools (booking, searching)
agent = HospitalAgent(rag_pipeline)

# 3. MOUTH: Text-to-Speech
tts = TextToSpeech()

# 4. EARS: Speech-to-Text
stt = SpeechToText()

# 5. AVATAR: D-ID Live Streaming (replaces HeyGen)
did_client: DIDClient | None = None
if settings.DID_API_KEY and settings.DID_SOURCE_URL:
    try:
        did_client = DIDClient(settings.DID_API_KEY)
        logger.info("✅ D-ID client ready (live streaming enabled)")
    except Exception as e:
        logger.error(f"❌ D-ID client init failed: {e}")
else:
    missing = []
    if not settings.DID_API_KEY:
        missing.append("DID_API_KEY")
    if not settings.DID_SOURCE_URL:
        missing.append("DID_SOURCE_URL")
    if missing:
        logger.info(f"ℹ️ D-ID not enabled (missing: {', '.join(missing)})")


# ==================== Request/Response Models ====================
#
# Pydantic models define the SHAPE of data that our API accepts and returns.
# This gives us:
# 1. Automatic validation (reject bad requests)
# 2. Auto-generated API documentation
# 3. Type hints for our IDE
#

class ChatRequest(BaseModel):
    """What the frontend sends to us."""
    question: str  # The patient's question
    
    # Example:
    # {"question": "What are your visiting hours?"}

class ChatResponse(BaseModel):
    """What we send back to the frontend."""
    answer: str           # The AI's response
    sources: list = []    # What hospital data was used (for transparency)
    
    # Example:
    # {"answer": "Visiting hours are 10 AM - 12 PM...", "sources": [...]}


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """
    Root endpoint — just confirms the API is running.
    Visit http://localhost:8000/ in your browser to see this.
    """
    return {
        "message": "🏥 Hospital Video Agent API is running!",
        "status": "online",
        "model": settings.LLM_MODEL,
        "docs": "Visit /docs for interactive API documentation"
    }


@app.get("/did/stream/start")
async def did_start_stream():
    """Start a D-ID streaming session and return WebRTC config for frontend."""
    if not did_client:
        return {"status": "error", "message": "D-ID not configured"}
    try:
        # Create streaming session
        result = did_client.create_stream(
            source_url=settings.DID_SOURCE_URL,
            voice_id=settings.DID_VOICE_ID
        )
        return {
            "status": "success",
            "session_id": result["session_id"],
            "stream_id": result["stream_id"],
            "sdp_offer": result["sdp_offer"],
            "ice_servers": result["ice_servers"],
        }
    except Exception as e:
        logger.error(f"D-ID stream start failed: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/did/stream/answer")
async def did_submit_answer(data: dict):
    """Submit WebRTC SDP answer from frontend to complete handshake."""
    if not did_client:
        return {"status": "error", "message": "D-ID not configured"}
    try:
        sdp_answer = data.get("sdp_answer", "")
        if not sdp_answer:
            return {"status": "error", "message": "sdp_answer required"}
        did_client.submit_sdp_answer(sdp_answer)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"D-ID answer submit failed: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/did/stream/ice")
async def did_submit_ice(data: dict):
    """Submit ICE candidate for NAT traversal."""
    if not did_client:
        return {"status": "error", "message": "D-ID not configured"}
    try:
        candidate = data.get("candidate", "")
        sdp_mid = data.get("sdpMid", "")
        sdp_mline_index = data.get("sdpMLineIndex", 0)
        did_client.submit_ice_candidate(candidate, sdp_mid, sdp_mline_index)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"D-ID ICE submit failed: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/did/stream/close-all")
async def did_close_all():
    """Manually close all D-ID sessions (use when max sessions reached)."""
    if not did_client:
        return {"status": "error", "message": "D-ID not configured"}
    try:
        did_client.close_all_sessions()
        return {"status": "success", "message": "All sessions closed"}
    except Exception as e:
        logger.error(f"D-ID close all failed: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/health")
async def health_check():
    """
    Health check endpoint — shows system status.
    
    WHY: Useful for monitoring. Frontend can ping this to check if backend is up.
    Also shows how many chunks are in the knowledge base.
    """
    stats = rag_pipeline.get_stats()
    return {
        "status": "healthy",
        "rag_status": stats["status"],
        "knowledge_base_chunks": stats["chunk_count"],
        "llm_model": stats.get("model", "unknown"),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    💬 Main chat endpoint — send a question, get an answer.
    
    This is the PRIMARY endpoint that our frontend will use.
    
    HOW it works:
    1. Frontend sends: {"question": "Do you have a cardiologist?"}
    2. This function calls rag_pipeline.query()
    3. RAG searches hospital data, finds cardiology info
    4. LLM generates a natural answer
    5. We return: {"answer": "Yes! Dr. Rajesh Sharma is our cardiologist..."}
    
    Try it: 
    - Open http://localhost:8000/docs
    - Find the POST /chat endpoint
    - Click "Try it out"
    - Enter: {"question": "What health packages do you offer?"}
    """
    logger.info(f"💬 Chat request: {request.question}")
    
    # Call the smart Agent (instead of the raw RAG pipeline)
    result = agent.process_message(request.question)
    
    return ChatResponse(
        answer=result["answer"],
        sources=[] # Agent abstracted sources away, though we could fetch from memory if needed
    )


@app.post("/reset-knowledge-base")
async def reset_knowledge_base():
    """
    🔄 Reset and rebuild the knowledge base from scratch.
    
    Use this if you update hospital_knowledge.json and want to re-index.
    """
    rag_pipeline.reset_knowledge_base()
    stats = rag_pipeline.get_stats()
    return {
        "status": "success",
        "message": "Knowledge base reset and rebuilt",
        "chunks": stats["chunk_count"]
    }


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    🔌 WebSocket endpoint for real-time chat.
    
    WHY WebSocket instead of HTTP?
    - HTTP: Client sends request → waits → gets response → connection closes
    - WebSocket: Connection stays open, both sides can send messages anytime
    
    For video chat, we need continuous back-and-forth communication,
    not one-at-a-time request-response. WebSocket gives us that.
    
    HOW this works:
    1. Client connects to ws://localhost:8000/ws/chat
    2. Connection stays open
    3. Client sends: {"type": "text", "message": "Hello"}
    4. Server processes and sends back: {"type": "response", "message": "Hi! How can I help?"}
    5. This continues until the client disconnects
    
    Later (Phase 3), we'll add "audio" message type for voice processing.
    """
    await websocket.accept()
    logger.info("🔌 WebSocket connection established")
    latest_request_id = 0
    active_response_task: asyncio.Task | None = None

    async def _send_text_to_did(text: str):
        """Send text to D-ID for live streaming avatar speech."""
        if did_client is None or not did_client.stream_id:
            return
        try:
            # Send text to D-ID - avatar speaks in real-time via WebRTC
            success = did_client.send_text(text)
            if success:
                logger.info(f"🎙️ Sent to D-ID live stream: {text[:50]}...")
            else:
                logger.warning("Failed to send text to D-ID")
        except Exception as e:
            logger.error(f"❌ D-ID send text failed: {e}")

    async def _process_and_respond(question: str, request_id: int, direct_text: bool = False):
        try:
            if direct_text:
                agent_text = question
            else:
                # Run blocking agent call in thread so websocket loop stays responsive.
                result = await asyncio.to_thread(agent.process_message, question)
                agent_text = result["answer"]

            # Ignore stale responses if user asked a newer question.
            if request_id != latest_request_id:
                logger.info(f"⏭️ Skipping stale response for request_id={request_id}")
                return

            await websocket.send_json({
                "type": "response",
                "message": agent_text,
                "request_id": request_id
            })

            # Send to D-ID stream if enabled; otherwise fallback TTS audio.
            if did_client is not None and did_client.stream_id:
                await _send_text_to_did(agent_text)
                return

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                tmp_out_path = tmp_out.name

            try:
                success = await asyncio.to_thread(tts.synthesize, agent_text, tmp_out_path)
                if success and os.path.exists(tmp_out_path):
                    with open(tmp_out_path, "rb") as f:
                        out_audio_bytes = f.read()
                    # Re-check staleness before sending generated audio.
                    if request_id == latest_request_id:
                        out_audio_b64 = base64.b64encode(out_audio_bytes).decode("utf-8")
                        await websocket.send_json({
                            "type": "audio_response",
                            "data": out_audio_b64,
                            "request_id": request_id
                        })
            finally:
                if os.path.exists(tmp_out_path):
                    os.unlink(tmp_out_path)

        except asyncio.CancelledError:
            logger.info(f"🛑 Cancelled response task for request_id={request_id}")
            raise
        except Exception as e:
            logger.error(f"❌ Response task failed for request_id={request_id}: {e}")
            await websocket.send_json({
                "type": "error",
                "message": "Sorry, I had trouble processing that.",
                "request_id": request_id
            })
    
    try:
        while True:
            # Wait for a JSON message from the client frontend
            data = await websocket.receive_json()
            message_type = data.get("type", "text")
            
            # 1. Extract the text question from either TEXT or AUDIO
            question = ""
            direct_text = False
            if message_type == "text":
                question = data.get("message", "")
                latest_request_id = int(data.get("request_id") or (latest_request_id + 1))
                logger.info(f"⌨️ Received Text: {question}")
            elif message_type == "greeting":
                question = data.get("message", "")
                latest_request_id = int(data.get("request_id") or (latest_request_id + 1))
                direct_text = True
                logger.info("👋 Received startup greeting request")
                
            elif message_type == "audio":
                # Audio comes in as Base64 encoded string
                audio_b64 = data.get("data", "")
                if audio_b64:
                    logger.info("🎧 Received Audio recording from patient!")
                    # Handle data URI prefix if frontend sends it (e.g. data:audio/wav;base64,.....)
                    if "," in audio_b64:
                        audio_b64 = audio_b64.split(",")[1]
                        
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    # Save to a temporary file for Whisper
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                        tmp_audio.write(audio_bytes)
                        tmp_audio_path = tmp_audio.name
                        
                    # Whisper Transcribe (EARS)
                    question = stt.transcribe(tmp_audio_path)
                    os.unlink(tmp_audio_path) # Clean up temp file
                    
                    # Echo back what we heard so the frontend can display it
                    await websocket.send_json({
                        "type": "transcript",
                        "message": question
                    })
            
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue
                
            # 2. Process the question through our Agent (BRAIN)
            if not question:
                continue

            if active_response_task and not active_response_task.done():
                active_response_task.cancel()

            active_response_task = asyncio.create_task(
                _process_and_respond(question, latest_request_id, direct_text=direct_text)
            )
            
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected. Clearing agent memory.")
        if active_response_task and not active_response_task.done():
            active_response_task.cancel()
        agent.reset_memory()
        # Cleanup D-ID stream
        if did_client and did_client.stream_id:
            did_client.delete_stream()
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")


# ==================== Run the Server ====================
#
# This runs when you execute: python main.py
# 
# uvicorn is the ASGI server that actually serves our FastAPI app.
# ASGI = Asynchronous Server Gateway Interface
#   - Supports async/await (handle many requests simultaneously)
#   - Supports WebSockets (for our video chat)
#
# reload=True means the server restarts automatically when you save code changes.
# (Only use in development, not production)
#
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",          # "filename:app_variable_name"
        host=settings.HOST,   # 0.0.0.0 = accessible from any IP
        port=settings.PORT,   # Port 8000
        reload=True           # Auto-restart on code changes (dev only)
    )