"""
Style presets for writing and visual content generation.
These are default styles available to all users without needing to be stored per user.
"""

from typing import Dict, List

# Preset writing styles with detailed descriptions for LLM prompting
PRESET_WRITING_STYLES: Dict[str, Dict[str, str]] = {
    "storyteller": {
        "name": "Storyteller",
        "description": "Engaging narrative style with vivid descriptions, emotional connection, and compelling story arcs. Uses metaphors, anecdotes, and conversational tone to captivate readers.",
        "icon": "📖"
    },
    "executive": {
        "name": "Executive",
        "description": "Concise, polished, and professional style focused on clarity and efficiency. Uses bullet points, clear hierarchies, and direct language suitable for business communications.",
        "icon": "💼"
    },
    "provocateur": {
        "name": "Provocateur",
        "description": "Bold, contrarian style that challenges conventional thinking. Uses strong statements, rhetorical questions, and unexpected perspectives to spark debate and engagement.",
        "icon": "⚡"
    },
    "analyst": {
        "name": "Analyst",
        "description": "Data-driven, methodical style emphasizing facts, statistics, and logical reasoning. Uses structured analysis, evidence-based arguments, and objective observations.",
        "icon": "📊"
    }
}

# Preset visual styles for image generation
PRESET_VISUAL_STYLES: Dict[str, Dict[str, str]] = {
    "photorealistic": {
        "name": "Photorealistic",
        "description": "High-quality photographic style with realistic lighting, textures, and details. Sharp focus, natural colors, professional photography aesthetic.",
        "icon": "📷"
    },
    "minimalist": {
        "name": "Minimalist",
        "description": "Clean, simple design with ample white space, geometric shapes, and limited color palette. Modern, uncluttered aesthetic focused on essential elements.",
        "icon": "⬜"
    },
    "editorial": {
        "name": "Editorial",
        "description": "Magazine-quality illustration style with bold graphics, strong composition, and sophisticated color schemes. Professional, polished, and visually striking.",
        "icon": "🎨"
    },
    "vibrant": {
        "name": "Vibrant",
        "description": "Bold, energetic style with saturated colors, dynamic composition, and high contrast. Eye-catching and expressive with creative flair.",
        "icon": "🌈"
    }
}

def get_preset_writing_styles() -> List[Dict[str, str]]:
    """Returns list of preset writing styles with metadata."""
    return [
        {
            "id": key,
            "name": value["name"],
            "description": value["description"],
            "icon": value["icon"],
            "is_preset": True
        }
        for key, value in PRESET_WRITING_STYLES.items()
    ]

def get_preset_visual_styles() -> List[Dict[str, str]]:
    """Returns list of preset visual styles with metadata."""
    return [
        {
            "id": key,
            "name": value["name"],
            "description": value["description"],
            "icon": value["icon"],
            "is_preset": True
        }
        for key, value in PRESET_VISUAL_STYLES.items()
    ]

def get_writing_style_description(style_id: str) -> str:
    """Get the description for a preset writing style."""
    return PRESET_WRITING_STYLES.get(style_id, {}).get("description", "Natural, clear writing style")

def get_visual_style_description(style_id: str) -> str:
    """Get the description for a preset visual style."""
    return PRESET_VISUAL_STYLES.get(style_id, {}).get("description", "Professional, high-quality visual style")
