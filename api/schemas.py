from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import datetime

class ChatRequest(BaseModel):
    problem: Optional[str] = None
    session_id: Optional[str] = None
    writing_style_id: Optional[str] = None  # ID of writing style (preset or custom)
    visual_style_id: Optional[str] = None   # ID of visual style (preset or custom)

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
# Style Preferences Schemas
# =====================

class StylePreference(BaseModel):
    id: str
    name: str
    description: str
    is_preset: bool
    icon: Optional[str] = None
    created_at: Optional[datetime] = None

class CreateStyleRequest(BaseModel):
    name: str
    description: str

class UpdateStyleRequest(BaseModel):
    name: str
    description: str

class UserPreferencesResponse(BaseModel):
    writing_styles: List[StylePreference]
    visual_styles: List[StylePreference]

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
