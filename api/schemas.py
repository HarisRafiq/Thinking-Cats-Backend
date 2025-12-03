from typing import Optional, Dict, Any
from pydantic import BaseModel
from modular_agent.config import DEFAULT_MODEL

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

class UpdateTierRequest(BaseModel):
    user_email: str
    tier: str

class ShareToggleRequest(BaseModel):
    is_shared: bool

class UpdateUserStatusRequest(BaseModel):
    is_blocked: bool

class UpdateUserTierRequest(BaseModel):
    tier: str
