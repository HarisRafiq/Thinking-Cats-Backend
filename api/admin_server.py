import os
import sys
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_agent.database import DatabaseManager

app = FastAPI(title="Admin Panel API")

# CORS - allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database Manager
db_manager = DatabaseManager()

@app.on_event("startup")
async def startup_event():
    await db_manager.connect()

@app.on_event("shutdown")
def shutdown_event():
    db_manager.close()

@app.get("/admin/stats")
async def get_admin_stats():
    """Get aggregate statistics for admin dashboard."""
    return await db_manager.get_admin_stats()

@app.get("/admin/users")
async def get_admin_users(
    limit: int = Query(50, ge=1, le=1000), 
    skip: int = Query(0, ge=0), 
    search: Optional[str] = Query(None, description="Search by email or name"),
    tier: Optional[str] = Query(None, description="Filter by subscription tier"),
    is_blocked: Optional[bool] = Query(None, description="Filter by blocked status"),
):
    """Get list of users with search and filter options."""
    return await db_manager.get_all_users(
        limit=limit, 
        skip=skip,
        search=search,
        tier=tier,
        is_blocked=is_blocked
    )

@app.get("/admin/sessions")
async def get_admin_sessions(
    limit: int = Query(50, ge=1, le=1000),
    skip: int = Query(0, ge=0),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[str] = Query(None, description="Filter by session status"),
    is_shared: Optional[bool] = Query(None, description="Filter by shared status"),
):
    """Get list of sessions with filter options."""
    return await db_manager.get_all_sessions_admin(
        limit=limit,
        skip=skip,
        user_id=user_id,
        status=status,
        is_shared=is_shared
    )

@app.get("/admin/logs")
async def get_admin_logs(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    model: Optional[str] = Query(None, description="Filter by model"),
    limit: int = Query(50, ge=1, le=1000), 
    skip: int = Query(0, ge=0), 
):
    """Get LLM logs with filter options."""
    return await db_manager.get_llm_logs(
        user_id=user_id,
        session_id=session_id,
        model=model,
        limit=limit, 
        skip=skip
    )

from pydantic import BaseModel
from datetime import datetime
from fastapi import HTTPException

class UpdateUserStatusRequest(BaseModel):
    is_blocked: bool

class UpdateUserTierRequest(BaseModel):
    tier: str

@app.put("/admin/users/{user_id}/status")
async def update_user_status(user_id: str, request: UpdateUserStatusRequest):
    """Update user blocked status."""
    success = await db_manager.set_user_status(user_id, request.is_blocked)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "updated", "is_blocked": request.is_blocked}

@app.put("/admin/users/{user_id}/tier")
async def update_user_tier_admin(user_id: str, request: UpdateUserTierRequest):
    """Update user tier (admin)."""
    success = await db_manager.set_user_tier(user_id, request.tier)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "updated", "tier": request.tier}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ADMIN_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)

