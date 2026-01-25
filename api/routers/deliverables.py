"""
Deliverables Router
API for premium block-based deliverables (social posts, reports, etc.) 
"""
import json
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from api.core_dependencies import get_db_manager, get_block_generator
from api.dependencies import get_current_user
from modular_agent.block_generator import (
    BlockGenerator,
    DeliverableGenerationError,
    ImageGenerationError,
)

router = APIRouter(prefix="/deliverables", tags=["deliverables"])

# =====================
# Request Schemas
# =====================

class SuggestionRequest(BaseModel):
    session_id: str

class GenerateRequest(BaseModel):
    session_id: str
    deliverable_type: str = "social"  # "social", "report", "document"
    title: Optional[str] = None
    instruction: Optional[str] = None
    writing_style_id: Optional[str] = None  # Override session style if provided
    visual_style_id: Optional[str] = None   # Override session style if provided

# =====================
# Endpoints
# =====================

@router.get("")
async def list_deliverables(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """List all deliverables for the current user."""
    deliverables = await db_manager.get_user_deliverables(user_id=str(current_user["_id"]))
    return {"deliverables": deliverables}

@router.post("/generate")
async def generate_deliverable(
    request: GenerateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager),
    block_gen: BlockGenerator = Depends(get_block_generator)
):
    """Generate a premium deliverable block-by-block via SSE."""
    user_id = str(current_user["_id"])

    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    
    # Check session ownership if provided
    if request.session_id:
        session = await db_manager.get_session(request.session_id)
        if not session or session.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
    else:
        session = None

    async def event_generator():
        try:
            writing_style_id = request.writing_style_id or (session or {}).get("writing_style_id")
            visual_style_id = request.visual_style_id or (session or {}).get("visual_style_id")

            title = request.title or f"New {request.deliverable_type.capitalize()}"

            async for update in block_gen.generate_deliverable(
                user_id=user_id,
                session_id=request.session_id,
                deliverable_type=request.deliverable_type,
                title=title,
                instruction=request.instruction,
                writing_style_id=writing_style_id,
                visual_style_id=visual_style_id
            ):
                event_name = update.get("event", "update")
                payload = {k: v for k, v in update.items() if k != "event"}
                yield {"event": event_name, "data": json.dumps(payload)}

        except DeliverableGenerationError as e:
            yield {
                "event": "error",
                "data": json.dumps({
                    "message": str(e),
                    "deliverable_id": getattr(e, "deliverable_id", None)
                })
            }
        except ImageGenerationError as e:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)})
            }
        except Exception as e:
            print(f"Error in deliverable generation: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"message": "Unexpected error during deliverable generation"})
            }

    return EventSourceResponse(event_generator())

@router.get("/{deliverable_id}")
async def get_deliverable(
    deliverable_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Retrieve a specific deliverable and its blocks."""
    deliverable = await db_manager.get_deliverable(deliverable_id)
    if not deliverable:
        raise HTTPException(status_code=404, detail="Deliverable not found")
    
    if deliverable.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return {"deliverable": deliverable}

@router.delete("/{deliverable_id}")
async def delete_deliverable(
    deliverable_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Delete a deliverable."""
    # Check ownership
    deliverable = await db_manager.get_deliverable(deliverable_id)
    if not deliverable:
        raise HTTPException(status_code=404, detail="Deliverable not found")
    
    if deliverable.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db_manager.delete_deliverable(deliverable_id)
    return {"status": "success"}


# =====================
# Block Operations
# =====================

class UpdateBlockRequest(BaseModel):
    content: Dict[str, Any]

class RegenerateBlockRequest(BaseModel):
    instruction: Optional[str] = None

@router.patch("/{deliverable_id}/blocks/{block_id}")
async def update_block(
    deliverable_id: str,
    block_id: str,
    request: UpdateBlockRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Update a specific block's content."""
    # Check ownership
    deliverable = await db_manager.get_deliverable(deliverable_id)
    if not deliverable:
        raise HTTPException(status_code=404, detail="Deliverable not found")
    
    if deliverable.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Verify block exists
    block = await db_manager.get_block(deliverable_id, block_id)
    if not block:
        raise HTTPException(status_code=404, detail="Block not found")
    
    success = await db_manager.update_block(deliverable_id, block_id, request.content, instruction="Manual Edit")
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update block")
    
    # Get updated block
    updated_block = await db_manager.get_block(deliverable_id, block_id)
    
    return {"status": "success", "block": updated_block}


@router.post("/{deliverable_id}/blocks/{block_id}/regenerate")
async def regenerate_block(
    deliverable_id: str,
    block_id: str,
    request: RegenerateBlockRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager),
    block_gen: BlockGenerator = Depends(get_block_generator)
):
    """Regenerate a specific block using AI."""
    # Check ownership
    deliverable = await db_manager.get_deliverable(deliverable_id)
    if not deliverable:
        raise HTTPException(status_code=404, detail="Deliverable not found")
    
    if deliverable.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    session_id = deliverable.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Deliverable has no linked session")
    
    try:
        new_block = await block_gen.regenerate_block(
            deliverable_id=deliverable_id,
            block_id=block_id,
            session_id=session_id,
            instruction=request.instruction
        )
        return {"status": "success", "block": new_block}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
