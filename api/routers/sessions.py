import os
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from api.schemas import ShareToggleRequest
from api.core_dependencies import get_db_manager, get_session_manager
from api.dependencies import get_current_user

router = APIRouter()

@router.get("/sessions")
async def list_sessions(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """List recent sessions."""
    return await db_manager.get_all_sessions(user_id=str(current_user["_id"]))

@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str, 
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager),
    session_manager = Depends(get_session_manager)
):
    """Get session with slides."""
    session = await db_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Get slides array, default to empty if not present
    slides = session.get("slides", [])
    
    # Convert any ObjectIds in slides to strings
    for slide in slides:
        if "id" in slide and not isinstance(slide["id"], str):
            slide["id"] = str(slide["id"])
    
    # Check if session is actively processing
    active_session = session_manager.get_session(session_id)
    is_processing = False
    if active_session:
        is_processing = active_session.get("is_processing", False)
    
    return {
        "session_id": session_id,
        "model": session.get("model"),
        "slides": slides,
        "is_processing": is_processing,
        "is_shared": session.get("is_shared", False),
        "pending_interaction": session.get("pending_interaction")
    }

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str, 
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager),
    session_manager = Depends(get_session_manager)
):
    """Archive a session."""
    session = await db_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db_manager.archive_session(session_id)
    
    # If active, maybe we should stop it?
    session_manager.cancel_session(session_id)
        
    return {"status": "archived"}

@router.post("/sessions/{session_id}/share")
async def toggle_session_sharing(
    session_id: str, 
    request: ShareToggleRequest, 
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Toggle sharing on/off for a session. Requires authentication and ownership."""
    # Verify session exists and ownership
    session = await db_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to access this session")
    
    # Toggle sharing
    success = await db_manager.toggle_session_sharing(session_id, request.is_shared)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update sharing status")
    
    # Generate public URL (assuming frontend domain from env or default)
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    public_url = f"{frontend_url}/share/{session_id}" if request.is_shared else None
    
    return {
        "session_id": session_id,
        "is_shared": request.is_shared,
        "public_url": public_url
    }

@router.get("/share/{session_id}")
async def get_shared_session(session_id: str, db_manager = Depends(get_db_manager)):
    """Get shared session data (public, no auth required)."""
    session = await db_manager.get_shared_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or not shared")
    
    # Get slides array, default to empty if not present
    slides = session.get("slides", [])
    
    # Convert any ObjectIds in slides to strings
    for slide in slides:
        if "id" in slide and not isinstance(slide["id"], str):
            slide["id"] = str(slide["id"])
    
    # Generate metadata for SEO
    first_message = None
    if session.get("messages") and len(session["messages"]) > 0:
        first_message = session["messages"][0].get("content", "")
    
    title = first_message[:100] if first_message else "Conversation on ThinkingCats"
    description = f"See how AI experts collaborate to solve: {first_message[:150]}" if first_message else "See how AI experts collaborate to solve complex problems on ThinkingCats"
    
    return {
        "session_id": session_id,
        "model": session.get("model"),
        "slides": slides,
        "problem": session.get("problem", ""),
        "created_at": session.get("created_at"),
        "metadata": {
            "title": title,
            "description": description
        }
    }
