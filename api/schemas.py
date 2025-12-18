from typing import Optional, Dict, Any, List
from pydantic import BaseModel

class ChatRequest(BaseModel):
    problem: Optional[str] = None
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

class UpdateTierRequest(BaseModel):
    user_email: str
    tier: str

class ShareToggleRequest(BaseModel):
    is_shared: bool

class UpdateUserStatusRequest(BaseModel):
    is_blocked: bool

class UpdateUserTierRequest(BaseModel):
    tier: str

# =====================
# NOTE: Artifact schemas are defined inline in routers/artifacts.py for simplicity
# The router uses: GenerateRequest, UpdateRequest
