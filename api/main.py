import sys
import os
import asyncio
import json
from datetime import timedelta
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_agent.orchestrator import Orchestrator
from modular_agent.config import DEFAULT_MODEL
from modular_agent.database import DatabaseManager
from api.events import BaseEvent, DoneEvent, ErrorEvent, ProcessingStartEvent, ProcessingCompleteEvent
from api.auth import verify_google_token, create_access_token, oauth2_scheme, SECRET_KEY, ALGORITHM
import jwt
from fastapi import Depends, status
from fastapi.security import OAuth2PasswordBearer

app = FastAPI(title="Multi-Agent Chat API")

# CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database Manager
db_manager = DatabaseManager()

# Store active sessions
# Key: session_id, Value: { 'output_queue': asyncio.Queue, 'input_queue': asyncio.Queue, 'task': asyncio.Task }
sessions: Dict[str, Dict[str, Any]] = {}

class ChatRequest(BaseModel):
    problem: Optional[str] = None
    model: str = DEFAULT_MODEL
    session_id: Optional[str] = None

class MessageRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    session_id: str

class AuthRequest(BaseModel):
    token: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: Dict[str, Any]

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Dependency to get the current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    user = await db_manager.get_user_by_id(user_id)
    if user is None:
        raise credentials_exception
        
    return user

async def activate_session(session_id: str, session_data: Dict[str, Any]) -> None:
    """
    Activate a session by creating queues and starting the orchestrator worker.
    Used for resuming sessions that exist in DB but not in memory.
    """
    if session_id in sessions:
        # Already active
        return
    
    print(f"Activating session {session_id}...")
    
    # Create queues for the session
    output_queue = asyncio.Queue(maxsize=1000)
    input_queue = asyncio.Queue(maxsize=100)
    
    sessions[session_id] = {
        "output_queue": output_queue,
        "input_queue": input_queue,
        "task": None,
        "is_processing": False
    }
    
    # Start the orchestrator worker (without initial problem since we're resuming)
    task = asyncio.create_task(
        orchestrator_worker(
            session_id, 
            problem=None,  # No initial problem when resuming
            model=session_data.get("model", DEFAULT_MODEL), 
            input_queue=input_queue, 
            output_queue=output_queue
        )
    )
    
    sessions[session_id]["task"] = task
    print(f"Session {session_id} activated successfully")

async def orchestrator_worker(session_id: str, problem: Optional[str], model: str, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
    """
    Async worker function that loops to process incoming tasks (solve or chat).
    """
    
    async def callback(event_data: Dict[str, Any]):
        """Put event into async queue with backpressure handling."""
        try:
            # Try to put event immediately (non-blocking if queue has space)
            output_queue.put_nowait(event_data)
        except asyncio.QueueFull:
            # Queue is full - drop oldest event and add new one (backpressure handling)
            try:
                # Remove oldest event
                output_queue.get_nowait()
                # Add new event
                output_queue.put_nowait(event_data)
                print(f"[Session {session_id}] Queue full, dropped oldest event")
            except asyncio.QueueEmpty:
                # Queue was emptied between checks, try again
                try:
                    output_queue.put_nowait(event_data)
                except asyncio.QueueFull:
                    # Still full, log and skip this event
                    print(f"[Session {session_id}] Queue persistently full, dropping event: {event_data.get('type')}")

    try:
        orchestrator = Orchestrator(
            model_name=model, 
            verbose=True, 
            event_callback=callback,
            session_id=session_id,
            db_manager=db_manager
        )
        
        # If resuming, we might want to load history here
        await orchestrator.load_session_history()
        
        # Initial process if problem is provided
        if problem:
            # Mark as processing
            if session_id in sessions:
                sessions[session_id]["is_processing"] = True
            # Save user message to DB (for agent context)
            await db_manager.add_message(session_id, {"role": "user", "content": problem})
            # Emit processing_start event
            processing_start_event = ProcessingStartEvent()
            await callback(processing_start_event.to_dict())
            await orchestrator.process(problem)
            # Emit processing_complete event
            processing_complete_event = ProcessingCompleteEvent()
            await callback(processing_complete_event.to_dict())
            # Mark as not processing
            if session_id in sessions:
                sessions[session_id]["is_processing"] = False
        
        # Event loop for follow-up messages
        while True:
            try:
                # Wait for next input
                # We use a timeout to check for cancellation or idle
                task = await asyncio.wait_for(input_queue.get(), timeout=600) # 10 minute timeout
                
                if task['type'] == 'message':
                    # Mark as processing
                    if session_id in sessions:
                        sessions[session_id]["is_processing"] = True
                    # Save user message to DB (for agent context)
                    await db_manager.add_message(session_id, {"role": "user", "content": task['content']})
                    
                    # Emit processing_start event
                    processing_start_event = ProcessingStartEvent()
                    await callback(processing_start_event.to_dict())
                    
                    # Note: user_message slide is created inside orchestrator.process()
                    await orchestrator.process(task['content'])
                    
                    # Emit processing_complete event
                    processing_complete_event = ProcessingCompleteEvent()
                    await callback(processing_complete_event.to_dict())
                    # Mark as not processing
                    if session_id in sessions:
                        sessions[session_id]["is_processing"] = False
                elif task['type'] == 'stop':
                    break
            except asyncio.TimeoutError:
                # Idle timeout
                print(f"Session {session_id} timed out")
                break
                
    except asyncio.CancelledError:
        print(f"Session {session_id} worker cancelled")
    except Exception as e:
        # Mark as not processing on error
        if session_id in sessions:
            sessions[session_id]["is_processing"] = False
        error_event = ErrorEvent(message=str(e))
        await output_queue.put(error_event.to_dict())
    finally:
        # Mark as not processing when worker exits
        if session_id in sessions:
            sessions[session_id]["is_processing"] = False
        # Signal end of stream
        done_event = DoneEvent()
        await output_queue.put(done_event.to_dict())
        
        # Clean up session
        if session_id in sessions:
            del sessions[session_id]
            print(f"Session {session_id} cleaned up after worker exit")

@app.on_event("startup")
async def startup_event():
    await db_manager.connect()

@app.on_event("shutdown")
def shutdown_event():
    db_manager.close()

@app.post("/auth/google", response_model=Token)
async def login_google(request: AuthRequest):
    # Validate request
    if not request.token:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Google authentication token is required",
                "code": "MISSING_TOKEN"
            }
        )
    
    # Verify Google Token
    try:
        google_user = verify_google_token(request.token)
        if not google_user:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid or expired Google authentication token",
                    "code": "INVALID_GOOGLE_TOKEN"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error during Google token verification: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error verifying Google token",
                "code": "TOKEN_VERIFICATION_ERROR"
            }
        )
    
    # Validate required user data
    if not google_user.get("email"):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Google token missing required user information",
                "code": "INCOMPLETE_USER_DATA"
            }
        )
    
    # Create or update user in DB
    try:
        user_data = {
            "email": google_user["email"],
            "name": google_user.get("name"),
            "picture": google_user.get("picture"),
            "google_id": google_user["sub"]
        }
        
        user_id = await db_manager.create_user(user_data)
        if not user_id:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to create user account",
                    "code": "USER_CREATION_FAILED"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Database error while creating user: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Database error while creating user account",
                "code": "DATABASE_ERROR"
            }
        )
    
    # Create JWT
    try:
        access_token_expires = timedelta(minutes=60 * 24 * 7) # 7 days
        access_token = create_access_token(
            data={"sub": user_id}, expires_delta=access_token_expires
        )
        if not access_token:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to create access token",
                    "code": "TOKEN_CREATION_FAILED"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating access token: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error creating access token",
                "code": "TOKEN_CREATION_ERROR"
            }
        )
    
    # Get full user object
    try:
        user = await db_manager.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "User created but could not be retrieved",
                    "code": "USER_RETRIEVAL_FAILED"
                }
            )
        
        user['id'] = str(user['_id'])
        del user['_id']
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving user: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error retrieving user information",
                "code": "USER_RETRIEVAL_ERROR"
            }
        )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user": user
    }

@app.post("/chat", response_model=ChatResponse)
async def start_chat(request: ChatRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    # If resuming an existing session
    if request.session_id:
        session_id = request.session_id
        
        # If session exists and is active, just return it
        if session_id in sessions:
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
        
    # If session exists in DB but not active, we are resuming
    # For now, we treat resuming same as starting new but with existing ID
    # The worker will handle loading if we implement it fully
    
    # Create queues with maxsize to prevent memory issues
    # maxsize=1000 allows buffering but prevents unbounded growth
    output_queue = asyncio.Queue(maxsize=1000)
    input_queue = asyncio.Queue(maxsize=100)
    
    sessions[session_id] = {
        "output_queue": output_queue,
        "input_queue": input_queue,
        "task": None,
        "is_processing": False
    }
    
    # Start the orchestrator in a background task
    task = asyncio.create_task(
        orchestrator_worker(session_id, request.problem if not request.session_id else None, request.model, input_queue, output_queue)
    )
    
    sessions[session_id]["task"] = task
    
    return ChatResponse(session_id=session_id)


@app.post("/chat/{session_id}/message")
async def send_message(session_id: str, request: MessageRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    # Check if session exists in memory
    if session_id not in sessions:
        # Check if it exists in database
        session = await db_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Verify ownership
        if session.get("user_id") != str(current_user["_id"]):
            raise HTTPException(status_code=403, detail="Not authorized to access this session")
        
        # Auto-activate the session
        await activate_session(session_id, session)
        
    await sessions[session_id]['input_queue'].put({
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
        sessions[session_id]['output_queue'].put_nowait(message_received_event)
    except asyncio.QueueFull:
        # Queue is full, but this is just an acknowledgment so it's okay to drop
        pass
    
    return {"status": "queued"}

@app.get("/stream/{session_id}")
async def stream_chat(session_id: str, token: str):
    # Manually verify token for SSE since EventSource doesn't support headers easily in all browsers
    # or we can use a library on frontend. For now, query param is easiest.
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
    await activate_session(session_id, session)
    
    event_queue = sessions[session_id]['output_queue']
    
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

@app.get("/sessions")
async def list_sessions(current_user: Dict[str, Any] = Depends(get_current_user)):
    """List recent sessions."""
    return await db_manager.get_all_sessions(user_id=str(current_user["_id"]))

@app.get("/sessions/{session_id}")
async def get_session(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get session with slides."""
    session = await db_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Get slides array, default to empty if not present
    slides = session.get("slides", [])
    
    # Convert any ObjectIds in slides to strings (though they should already be strings from add_slide)
    # This is a safety measure in case slides were added differently
    for slide in slides:
        if "id" in slide and not isinstance(slide["id"], str):
            slide["id"] = str(slide["id"])
    
    # Check if session is actively processing
    # Use the is_processing flag from the session dict
    is_processing = False
    if session_id in sessions:
        is_processing = sessions[session_id].get("is_processing", False)
    
    return {
        "session_id": session_id,
        "model": session.get("model"),
        "slides": slides,
        "is_processing": is_processing
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Archive a session."""
    session = await db_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db_manager.archive_session(session_id)
    
    # If active, maybe we should stop it?
    if session_id in sessions:
        # Cancel the task
        task = sessions[session_id].get("task")
        if task:
            task.cancel()
        del sessions[session_id]
        
    return {"status": "archived"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
