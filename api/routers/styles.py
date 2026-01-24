"""
API router for managing user style preferences (writing and visual styles).
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from api.schemas import (
    CreateStyleRequest, 
    UpdateStyleRequest, 
    UserPreferencesResponse
)
from api.dependencies import get_current_user
from api.core_dependencies import get_db_manager
from modular_agent.database import DatabaseManager
from modular_agent.style_presets import (
    get_preset_writing_styles,
    get_preset_visual_styles,
    PRESET_WRITING_STYLES,
    PRESET_VISUAL_STYLES
)

router = APIRouter(prefix="/styles", tags=["styles"])

@router.get("/", response_model=UserPreferencesResponse)
async def get_all_styles(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    Get all available styles (presets + user's custom styles).
    Returns writing and visual styles merged together.
    """
    user_id = str(current_user["_id"])
    
    # Get user's custom styles
    prefs = await db_manager.get_user_preferences(user_id)
    
    # Get preset styles
    preset_writing = get_preset_writing_styles()
    preset_visual = get_preset_visual_styles()
    
    # Merge preset + custom styles
    writing_styles = preset_writing + prefs.get("writing_styles", [])
    visual_styles = preset_visual + prefs.get("visual_styles", [])
    
    return UserPreferencesResponse(
        writing_styles=writing_styles,
        visual_styles=visual_styles
    )

@router.post("/writing", response_model=Dict[str, str])
async def create_writing_style(
    request: CreateStyleRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Create a new custom writing style."""
    user_id = str(current_user["_id"])
    
    # Validate input
    if not request.name or len(request.name.strip()) < 2:
        raise HTTPException(status_code=400, detail="Style name must be at least 2 characters")
    
    if not request.description or len(request.description.strip()) < 10:
        raise HTTPException(status_code=400, detail="Style description must be at least 10 characters")
    
    if len(request.description) > 500:
        raise HTTPException(status_code=400, detail="Style description must be less than 500 characters")
    
    # Check if user already has too many custom styles (limit: 20)
    prefs = await db_manager.get_user_preferences(user_id)
    if len(prefs.get("writing_styles", [])) >= 20:
        raise HTTPException(status_code=400, detail="Maximum 20 custom writing styles allowed")
    
    style_id = await db_manager.add_writing_style(user_id, request.name, request.description)
    
    return {"id": style_id, "message": "Writing style created successfully"}

@router.post("/visual", response_model=Dict[str, str])
async def create_visual_style(
    request: CreateStyleRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Create a new custom visual style."""
    user_id = str(current_user["_id"])
    
    # Validate input
    if not request.name or len(request.name.strip()) < 2:
        raise HTTPException(status_code=400, detail="Style name must be at least 2 characters")
    
    if not request.description or len(request.description.strip()) < 10:
        raise HTTPException(status_code=400, detail="Style description must be at least 10 characters")
    
    if len(request.description) > 500:
        raise HTTPException(status_code=400, detail="Style description must be less than 500 characters")
    
    # Check if user already has too many custom styles (limit: 20)
    prefs = await db_manager.get_user_preferences(user_id)
    if len(prefs.get("visual_styles", [])) >= 20:
        raise HTTPException(status_code=400, detail="Maximum 20 custom visual styles allowed")
    
    style_id = await db_manager.add_visual_style(user_id, request.name, request.description)
    
    return {"id": style_id, "message": "Visual style created successfully"}

@router.put("/writing/{style_id}", response_model=Dict[str, str])
async def update_writing_style(
    style_id: str,
    request: UpdateStyleRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Update a custom writing style. Cannot update preset styles."""
    user_id = str(current_user["_id"])
    
    # Check if it's a preset style
    if style_id in PRESET_WRITING_STYLES:
        raise HTTPException(status_code=403, detail="Cannot modify preset styles")
    
    # Validate input
    if not request.name or len(request.name.strip()) < 2:
        raise HTTPException(status_code=400, detail="Style name must be at least 2 characters")
    
    if not request.description or len(request.description.strip()) < 10:
        raise HTTPException(status_code=400, detail="Style description must be at least 10 characters")
    
    if len(request.description) > 500:
        raise HTTPException(status_code=400, detail="Style description must be less than 500 characters")
    
    success = await db_manager.update_writing_style(user_id, style_id, request.name, request.description)
    
    if not success:
        raise HTTPException(status_code=404, detail="Style not found or cannot be modified")
    
    return {"message": "Writing style updated successfully"}

@router.put("/visual/{style_id}", response_model=Dict[str, str])
async def update_visual_style(
    style_id: str,
    request: UpdateStyleRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Update a custom visual style. Cannot update preset styles."""
    user_id = str(current_user["_id"])
    
    # Check if it's a preset style
    if style_id in PRESET_VISUAL_STYLES:
        raise HTTPException(status_code=403, detail="Cannot modify preset styles")
    
    # Validate input
    if not request.name or len(request.name.strip()) < 2:
        raise HTTPException(status_code=400, detail="Style name must be at least 2 characters")
    
    if not request.description or len(request.description.strip()) < 10:
        raise HTTPException(status_code=400, detail="Style description must be at least 10 characters")
    
    if len(request.description) > 500:
        raise HTTPException(status_code=400, detail="Style description must be less than 500 characters")
    
    success = await db_manager.update_visual_style(user_id, style_id, request.name, request.description)
    
    if not success:
        raise HTTPException(status_code=404, detail="Style not found or cannot be modified")
    
    return {"message": "Visual style updated successfully"}

@router.delete("/writing/{style_id}", response_model=Dict[str, str])
async def delete_writing_style(
    style_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Delete a custom writing style. Cannot delete preset styles."""
    user_id = str(current_user["_id"])
    
    # Check if it's a preset style
    if style_id in PRESET_WRITING_STYLES:
        raise HTTPException(status_code=403, detail="Cannot delete preset styles")
    
    success = await db_manager.delete_writing_style(user_id, style_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Style not found or cannot be deleted")
    
    return {"message": "Writing style deleted successfully"}

@router.delete("/visual/{style_id}", response_model=Dict[str, str])
async def delete_visual_style(
    style_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Delete a custom visual style. Cannot delete preset styles."""
    user_id = str(current_user["_id"])
    
    # Check if it's a preset style
    if style_id in PRESET_VISUAL_STYLES:
        raise HTTPException(status_code=403, detail="Cannot delete preset styles")
    
    success = await db_manager.delete_visual_style(user_id, style_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Style not found or cannot be deleted")
    
    return {"message": "Visual style deleted successfully"}

@router.get("/writing/{style_id}/description")
async def get_writing_style_description(
    style_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Get the full description of a writing style (preset or custom)."""
    user_id = str(current_user["_id"])
    
    # Check if it's a preset
    if style_id in PRESET_WRITING_STYLES:
        return {
            "id": style_id,
            "description": PRESET_WRITING_STYLES[style_id]["description"]
        }
    
    # Check custom styles
    style = await db_manager.get_writing_style(user_id, style_id)
    if not style:
        raise HTTPException(status_code=404, detail="Style not found")
    
    return {
        "id": style_id,
        "description": style["description"]
    }

@router.get("/visual/{style_id}/description")
async def get_visual_style_description_endpoint(
    style_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Get the full description of a visual style (preset or custom)."""
    user_id = str(current_user["_id"])
    
    # Check if it's a preset
    if style_id in PRESET_VISUAL_STYLES:
        return {
            "id": style_id,
            "description": PRESET_VISUAL_STYLES[style_id]["description"]
        }
    
    # Check custom styles
    style = await db_manager.get_visual_style(user_id, style_id)
    if not style:
        raise HTTPException(status_code=404, detail="Style not found")
    
    return {
        "id": style_id,
        "description": style["description"]
    }
