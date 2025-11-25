"""
Event type definitions for the multi-agent chat system.
All events are serializable to JSON for SSE transmission.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json


class BaseEvent:
    """Base class for all events."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class ConsultStartEvent(BaseEvent):
    """Event emitted when orchestrator starts consulting an expert."""
    expert: str  # Real expert name (e.g., "Steve Jobs") - used by backend
    fictional_name: str  # Fictional display name (e.g., "Steve Paws") - used by frontend
    question: str
    type: str = "consult_start"
    one_liner: Optional[str] = None


@dataclass
class ConsultEndEvent(BaseEvent):
    """Event emitted when expert consultation completes."""
    expert: str  # Real expert name - used by backend
    fictional_name: str  # Fictional display name - used by frontend
    response: str
    type: str = "consult_end"
    one_liner: Optional[str] = None  # Include for consistency and easy access


@dataclass
class ErrorEvent(BaseEvent):
    """Event emitted when an error occurs."""
    message: str
    type: str = "error"
    expert: Optional[str] = None


@dataclass
class ClarificationRequestEvent(BaseEvent):
    """Event emitted when orchestrator needs clarification from user."""
    question: str
    type: str = "clarification_request"


@dataclass
class DoneEvent(BaseEvent):
    """Internal event to signal stream completion."""
    type: str = "done"


@dataclass
class SlideAddedEvent(BaseEvent):
    """Event emitted when a new slide is added to the session."""
    slide: Dict[str, Any]  # Full slide data including id
    type: str = "slide_added"


@dataclass
class OrchestratorThinkingEvent(BaseEvent):
    """Event emitted when the orchestrator is thinking/processing."""
    label: str = "Thinking..."
    type: str = "orchestrator_thinking"


# Event type registry for type-safe event creation
EVENT_TYPES = {
    "consult_start": ConsultStartEvent,
    "consult_end": ConsultEndEvent,
    "error": ErrorEvent,
    "clarification_request": ClarificationRequestEvent,
    "done": DoneEvent,
    "slide_added": SlideAddedEvent,
    "orchestrator_thinking": OrchestratorThinkingEvent,
}


def create_event(event_type: str, **kwargs) -> BaseEvent:
    """Factory function to create typed events."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event type: {event_type}")
    
    event_class = EVENT_TYPES[event_type]
    return event_class(**kwargs)

