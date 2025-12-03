import asyncio
import json
import jwt
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from sse_starlette.sse import EventSourceResponse
from api.schemas import ChatRequest, MessageRequest, ChatResponse
from api.core_dependencies import get_db_manager, get_session_manager
from api.dependencies import get_current_user
from api.auth import SECRET_KEY, ALGORITHM

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def start_chat(
    request: ChatRequest, 
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager),
    session_manager = Depends(get_session_manager)
):
    # Check user limits
    can_send = await db_manager.check_user_limit(str(current_user["_id"]))
    if not can_send:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Daily message limit reached. Please upgrade to Pro or wait until tomorrow.",
                "code": "LIMIT_REACHED"
            }
        )

    # If resuming an existing session
    if request.session_id:
        session_id = request.session_id
        
        # If session exists and is active, just return it
        if session_manager.get_session(session_id):
            return ChatResponse(session_id=session_id)
            
        # Verify ownership if resuming from DB
        session = await db_manager.get_session(session_id)
        if not session:
             raise HTTPException(status_code=404, detail="Session not found")
        if session.get("user_id") != str(current_user["_id"]):
             raise HTTPException(status_code=403, detail="Not authorized to access this session")
             
    else:
        # Create new session in MongoDB first to get the _id
        session_id = await db_manager.create_session({
            "problem": request.problem,
            "model": request.model
        }, user_id=str(current_user["_id"]))
        
    # Start new session via manager
    # Note: start_new_session handles both new and resuming (if we treat resuming as starting worker)
    # But wait, start_new_session in session_manager takes problem.
    # If resuming, problem is None.
    
    # Actually, session_manager has activate_session for resuming and start_new_session for new.
    # But start_new_session logic in my implementation was:
    # async def start_new_session(self, session_id: str, problem: str, model: str, user_id: str):
    
    if request.session_id:
         # Resuming logic
         # We need session data to know model
         session = await db_manager.get_session(session_id)
         await session_manager.activate_session(session_id, session)
    else:
         # New session logic
         await session_manager.start_new_session(
             session_id, 
             request.problem, 
             request.model, 
             str(current_user["_id"])
         )
    
    return ChatResponse(session_id=session_id)


@router.post("/chat/{session_id}/message")
async def send_message(
    session_id: str, 
    request: MessageRequest, 
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager),
    session_manager = Depends(get_session_manager)
):
    # Check if session exists in memory
    active_session = session_manager.get_session(session_id)
    if not active_session:
        # Check if it exists in database
        session = await db_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Verify ownership
        if session.get("user_id") != str(current_user["_id"]):
            raise HTTPException(status_code=403, detail="Not authorized to access this session")
        
        # Auto-activate the session
        await session_manager.activate_session(session_id, session)
        active_session = session_manager.get_session(session_id)
        
    # Check user limits
    can_send = await db_manager.check_user_limit(str(current_user["_id"]))
    if not can_send:
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Daily message limit reached. Please upgrade to Pro or wait until tomorrow.",
                "code": "LIMIT_REACHED"
            }
        )
        
    await active_session['input_queue'].put({
        "type": "message",
        "content": request.message,
        "timestamp": asyncio.get_event_loop().time()
    })
    
    # Emit immediate acknowledgment event
    import time
    message_received_event = {
        "type": "message_received",
        "message": request.message,
        "timestamp": time.time()
    }
    try:
        active_session['output_queue'].put_nowait(message_received_event)
    except asyncio.QueueFull:
        # Queue is full, but this is just an acknowledgment so it's okay to drop
        pass
    
    return {"status": "queued"}

@router.get("/stream/{session_id}")
async def stream_chat(
    session_id: str, 
    token: str,
    db_manager = Depends(get_db_manager),
    session_manager = Depends(get_session_manager)
):
    # Manually verify token for SSE since EventSource doesn't support headers easily in all browsers
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
        
    # Verify session ownership
    session = await db_manager.get_session(session_id)
    if not session:
         raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id") != user_id:
         raise HTTPException(status_code=403, detail="Not authorized")

    # Auto-activate session if it's not in memory
    active_session = session_manager.get_session(session_id)
    if not active_session:
        await session_manager.activate_session(session_id, session)
        active_session = session_manager.get_session(session_id)
    
    event_queue = active_session['output_queue']
    
    async def event_generator():
        """Optimized event generator with backpressure handling."""
        dropped_events = 0
        while True:
            try:
                # Wait for event with shorter timeout for better responsiveness
                # Use wait_for to allow periodic health checks
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield {"event": "ping", "data": json.dumps({"type": "heartbeat"})}
                    continue
                
                # Check if this is a done event
                if event.get("type") == "done":
                    yield {"event": "end", "data": json.dumps({"message": "Stream finished"})}
                    break
                
                event_type = event.get("type", "message")
                yield {"event": event_type, "data": json.dumps(event)}
                
                # Reset dropped events counter on successful send
                dropped_events = 0
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_data = {"type": "error", "message": str(e)}
                yield {"event": "error", "data": json.dumps(error_data)}
                break

    return EventSourceResponse(event_generator())
