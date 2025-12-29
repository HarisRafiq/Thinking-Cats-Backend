"""  
Unified Deliverables Router
Handles both document and social post generation with block-based structure.
"""
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from api.core_dependencies import get_db_manager, get_llm_provider, get_image_generator
from api.dependencies import get_current_user
from modular_agent.block_generator import BlockGenerator
from modular_agent.llm import GeminiProvider
from modular_agent.image_generator import ImageGenerator
import json

router = APIRouter(prefix="/deliverables", tags=["deliverables"])


# =====================
# Request Schemas
# =====================

class SuggestionsRequest(BaseModel):
    """Request deliverable suggestions from session."""
    session_id: str
    regenerate: bool = False  # Set to True to bypass cache and generate new suggestions


class GenerateRequest(BaseModel):
    """Request to generate new deliverable."""
    session_id: str
    deliverable_type: str  # "document", "social", "report"
    title: str
    custom_prompt: Optional[str] = None


# =====================
# Dependencies
# =====================

def get_block_generator(
    db_manager = Depends(get_db_manager),
    llm_provider: GeminiProvider = Depends(get_llm_provider),
    image_service: ImageGenerator = Depends(get_image_generator)
) -> BlockGenerator:
    """Get block generator instance."""
    return BlockGenerator(db_manager, llm_provider, image_service)


# =====================
# Endpoints
# =====================

@router.post("/suggestions")
async def get_suggestions(
    request: SuggestionsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    block_generator: BlockGenerator = Depends(get_block_generator)
):
    """Get suggested deliverables for a session. Uses cache unless regenerate=true."""
    try:
        suggestions = await block_generator.suggest_deliverables(
            request.session_id,
            regenerate=request.regenerate
        )
        return {"suggestions": suggestions, "cached": not request.regenerate}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_deliverable(
    request: GenerateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    block_generator: BlockGenerator = Depends(get_block_generator)
):
    """Generate deliverable with streaming progress (SSE)."""
    
    async def event_generator():
        async for event in block_generator.generate_deliverable(
            user_id=str(current_user["_id"]),
            session_id=request.session_id,
            deliverable_type=request.deliverable_type,
            title=request.title,
            custom_prompt=request.custom_prompt
        ):
            yield json.dumps(event)
    
    return EventSourceResponse(event_generator())


@router.get("")
async def list_deliverables(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """List all deliverables for current user."""
    deliverables = await db_manager.get_user_deliverables(user_id=str(current_user["_id"]))
    return {"deliverables": deliverables}


@router.get("/{deliverable_id}")
async def get_deliverable(
    deliverable_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Get specific deliverable with all blocks."""
    deliverable = await db_manager.get_deliverable(deliverable_id)
    if not deliverable:
        raise HTTPException(status_code=404, detail="Deliverable not found")
    if deliverable.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    return deliverable


@router.delete("/{deliverable_id}")
async def delete_deliverable(
    deliverable_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Delete deliverable."""
    deliverable = await db_manager.get_deliverable(deliverable_id)
    if not deliverable:
        raise HTTPException(status_code=404, detail="Deliverable not found")
    if deliverable.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db_manager.delete_deliverable(deliverable_id)
    return {"message": "Deliverable deleted"}
