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
# Social Platform Optimization Schemas
# =====================

class PlatformVariant(BaseModel):
    platform: str  # 'twitter', 'linkedin', 'instagram'
    caption: str
    hashtags: List[str]
    character_count: int
    image_url: Optional[str] = None  # Platform-optimized image URL
    aspect_ratio: str  # '1:1', '16:9', '2:1'

class CardOptimization(BaseModel):
    card_index: int
    original_url: str
    reading_sentence: str
    platform_variants: List[PlatformVariant]

class OptimizationRequest(BaseModel):
    generation_id: str

class OptimizationResponse(BaseModel):
    generation_id: str
    cards: List[CardOptimization]

# =====================
# NOTE: Artifact schemas are defined inline in routers/artifacts.py for simplicity
# The router uses: GenerateRequest, UpdateRequest
