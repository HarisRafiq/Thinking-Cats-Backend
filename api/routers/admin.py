from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from api.schemas import UpdateTierRequest, UpdateUserStatusRequest, UpdateUserTierRequest
from api.core_dependencies import get_db_manager
from api.dependencies import get_current_user

router = APIRouter()

@router.post("/admin/tier")
async def update_user_tier(
    request: UpdateTierRequest, 
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """
    Updates a user's subscription tier.
    For now, any authenticated user can do this (for demo purposes).
    In production, check for admin role.
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

@router.get("/admin/stats")
async def get_admin_stats(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Get aggregate statistics for admin dashboard."""
    # TODO: Add role check
    return await db_manager.get_admin_stats()

@router.get("/admin/users")
async def get_admin_users(
    limit: int = 50, 
    skip: int = 0, 
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Get list of users."""
    # TODO: Add role check
    return await db_manager.get_all_users(limit=limit, skip=skip)

@router.get("/admin/logs")
async def get_admin_logs(
    user_id: Optional[str] = None, 
    limit: int = 50, 
    skip: int = 0, 
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Get LLM logs."""
    # TODO: Add role check
    return await db_manager.get_llm_logs(user_id=user_id, limit=limit, skip=skip)

@router.put("/admin/users/{user_id}/status")
async def update_user_status(
    user_id: str, 
    request: UpdateUserStatusRequest, 
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Update user blocked status."""
    # TODO: Add role check
    success = await db_manager.set_user_status(user_id, request.is_blocked)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "updated", "is_blocked": request.is_blocked}

@router.put("/admin/users/{user_id}/tier")
async def update_user_tier_admin(
    user_id: str, 
    request: UpdateUserTierRequest, 
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Update user tier (admin)."""
    # TODO: Add role check
    success = await db_manager.set_user_tier(user_id, request.tier)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "updated", "tier": request.tier}
