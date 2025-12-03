from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from api.schemas import UpdateTierRequest, UpdateUserStatusRequest, UpdateUserTierRequest
from api.core_dependencies import get_db_manager
from api.dependencies import verify_admin_token

router = APIRouter()

@router.options("/admin/tier")
async def options_admin_tier():
    """Handle CORS preflight for admin tier update."""
    return {}

@router.post("/admin/tier")
async def update_user_tier(
    request: UpdateTierRequest, 
    _: bool = Depends(verify_admin_token),
    db_manager = Depends(get_db_manager)
):
    """
    Updates a user's subscription tier.
    Requires admin token authentication.
    """
    # Check if target user exists
    target_user = await db_manager.get_user_by_email(request.user_email)
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Update tier
    await db_manager.db.users.update_one(
        {"email": request.user_email},
        {"$set": {"subscription_tier": request.tier, "updated_at": datetime.utcnow()}}
    )
    
    return {"status": "updated", "email": request.user_email, "new_tier": request.tier}

@router.options("/admin/stats")
async def options_admin_stats():
    """Handle CORS preflight for admin stats."""
    return {}

@router.get("/admin/stats")
async def get_admin_stats(
    _: bool = Depends(verify_admin_token),
    db_manager = Depends(get_db_manager)
):
    """Get aggregate statistics for admin dashboard."""
    return await db_manager.get_admin_stats()

@router.options("/admin/users")
async def options_admin_users():
    """Handle CORS preflight for admin users."""
    return {}

@router.get("/admin/users")
async def get_admin_users(
    limit: int = Query(50, ge=1, le=1000), 
    skip: int = Query(0, ge=0), 
    search: Optional[str] = Query(None, description="Search by email or name"),
    tier: Optional[str] = Query(None, description="Filter by subscription tier"),
    is_blocked: Optional[bool] = Query(None, description="Filter by blocked status"),
    _: bool = Depends(verify_admin_token),
    db_manager = Depends(get_db_manager)
):
    """Get list of users with search and filter options."""
    return await db_manager.get_all_users(
        limit=limit, 
        skip=skip,
        search=search,
        tier=tier,
        is_blocked=is_blocked
    )

@router.options("/admin/sessions")
async def options_admin_sessions():
    """Handle CORS preflight for admin sessions."""
    return {}

@router.get("/admin/sessions")
async def get_admin_sessions(
    limit: int = Query(50, ge=1, le=1000),
    skip: int = Query(0, ge=0),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[str] = Query(None, description="Filter by session status"),
    is_shared: Optional[bool] = Query(None, description="Filter by shared status"),
    _: bool = Depends(verify_admin_token),
    db_manager = Depends(get_db_manager)
):
    """Get list of sessions with filter options."""
    return await db_manager.get_all_sessions_admin(
        limit=limit,
        skip=skip,
        user_id=user_id,
        status=status,
        is_shared=is_shared
    )

@router.options("/admin/logs")
async def options_admin_logs():
    """Handle CORS preflight for admin logs."""
    return {}

@router.get("/admin/logs")
async def get_admin_logs(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    model: Optional[str] = Query(None, description="Filter by model"),
    limit: int = Query(50, ge=1, le=1000), 
    skip: int = Query(0, ge=0), 
    _: bool = Depends(verify_admin_token),
    db_manager = Depends(get_db_manager)
):
    """Get LLM logs with filter options."""
    return await db_manager.get_llm_logs(
        user_id=user_id,
        session_id=session_id,
        model=model,
        limit=limit, 
        skip=skip
    )

@router.options("/admin/users/{user_id}/status")
async def options_admin_user_status():
    """Handle CORS preflight for admin user status update."""
    return {}

@router.put("/admin/users/{user_id}/status")
async def update_user_status(
    user_id: str, 
    request: UpdateUserStatusRequest, 
    _: bool = Depends(verify_admin_token),
    db_manager = Depends(get_db_manager)
):
    """Update user blocked status."""
    success = await db_manager.set_user_status(user_id, request.is_blocked)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "updated", "is_blocked": request.is_blocked}

@router.options("/admin/users/{user_id}/tier")
async def options_admin_user_tier():
    """Handle CORS preflight for admin user tier update."""
    return {}

@router.put("/admin/users/{user_id}/tier")
async def update_user_tier_admin(
    user_id: str, 
    request: UpdateUserTierRequest, 
    _: bool = Depends(verify_admin_token),
    db_manager = Depends(get_db_manager)
):
    """Update user tier (admin)."""
    success = await db_manager.set_user_tier(user_id, request.tier)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "updated", "tier": request.tier}
