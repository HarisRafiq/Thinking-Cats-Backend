"""
Context Manager for managing conversation history and building API contexts.
Simplifies context management by handling message storage, filtering, and API formatting.
"""
from typing import List, Dict, Any, Optional
from google.generativeai import protos


class ContextManager:
    """Manages conversation context with filtering and summarization capabilities."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # Store messages as simple dicts: {"role": "user|model", "content": str, ...}
        self.messages: List[Dict[str, Any]] = []
    
    def add_user_message(self, content: str):
        """Add a user message to the context."""
        self.messages.append({"role": "user", "content": content})
    
    def add_model_message(self, content: str, function_call: Optional[Any] = None):
        """Add a model message to the context."""
        msg = {"role": "model", "content": content}
        if function_call:
            msg["function_call"] = function_call
        self.messages.append(msg)
    
    def add_function_response(self, function_response: Any):
        """Add a function response (as user message in Gemini's format)."""
        self.messages.append({
            "role": "user",
            "content": "",
            "function_response": function_response
        })
    
    def get_history(self, 
                   include_function_calls: bool = True,
                   include_function_responses: bool = True,
                   max_messages: Optional[int] = None,
                   keep_first: bool = False,
                   keep_last: bool = True) -> List[Dict[str, Any]]:
        """
        Get filtered history based on criteria.
        
        Args:
            include_function_calls: Whether to include function call messages
            include_function_responses: Whether to include function response messages
            max_messages: Maximum number of messages to return (None = all)
            keep_first: If True, always keep the first message
            keep_last: If True, always keep the last message
            
        Returns:
            Filtered list of messages
        """
        filtered = []
        
        for i, msg in enumerate(self.messages):
            # Check if we should include this message
            should_include = True
            
            # Filter function calls
            if not include_function_calls and "function_call" in msg:
                should_include = False
            
            # Filter function responses
            if not include_function_responses and "function_response" in msg:
                should_include = False
            
            # Always keep first/last if requested
            if keep_first and i == 0:
                should_include = True
            if keep_last and i == len(self.messages) - 1:
                should_include = True
            
            if should_include:
                filtered.append(msg)
        
        # Apply max_messages limit
        if max_messages is not None and len(filtered) > max_messages:
            if keep_first and keep_last and max_messages >= 2:
                # Keep first and last, take middle messages
                first = filtered[0]
                last = filtered[-1]
                middle_count = max_messages - 2
                if middle_count > 0:
                    middle = filtered[1:-1][-middle_count:]
                    filtered = [first] + middle + [last]
                else:
                    filtered = [first, last]
            elif keep_last:
                # Keep last N messages
                filtered = filtered[-max_messages:]
            else:
                # Keep first N messages
                filtered = filtered[:max_messages]
        
        return filtered
    
    def get_final_response(self) -> Optional[str]:
        """Get the final model response (text only, no function calls)."""
        # Look backwards for the last text response
        for msg in reversed(self.messages):
            if msg.get("role") == "model" and msg.get("content") and "function_call" not in msg:
                return msg.get("content")
        return None
    
    def build_api_context(self, include_current_user: bool = True, current_user_message: str = "") -> List[protos.Content]:
        """
        Build context for Gemini API from stored messages.
        
        Args:
            include_current_user: Whether to include current_user_message in the context
            current_user_message: Current user message to add if include_current_user is True
            
        Returns:
            List of Content objects for generate_content()
        """
        contents = []
        
        # Add all stored messages
        for msg in self.messages:
            role = msg.get("role", "user")
            content_text = msg.get("content", "")
            
            if role == "user":
                if "function_response" in msg:
                    # Function response
                    fn_resp = msg["function_response"]
                    contents.append(
                        protos.Content(
                            role="user",
                            parts=[protos.Part(function_response=fn_resp)]
                        )
                    )
                else:
                    # Regular text message
                    contents.append(
                        protos.Content(
                            role="user",
                            parts=[protos.Part(text=content_text)]
                        )
                    )
            elif role == "model":
                if "function_call" in msg:
                    # Model message with function call
                    fn_call = msg["function_call"]
                    contents.append(
                        protos.Content(
                            role="model",
                            parts=[protos.Part(function_call=fn_call)]
                        )
                    )
                else:
                    # Regular text response
                    contents.append(
                        protos.Content(
                            role="model",
                            parts=[protos.Part(text=content_text)]
                        )
                    )
        
        # Add current user message if requested
        if include_current_user and current_user_message:
            contents.append(
                protos.Content(
                    role="user",
                    parts=[protos.Part(text=current_user_message)]
                )
            )
        
        return contents
    
    def clear(self):
        """Clear all messages."""
        self.messages = []
    
    def __len__(self):
        """Return the number of messages."""
        return len(self.messages)

