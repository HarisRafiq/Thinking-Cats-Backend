import inspect
from typing import Callable, List, Dict, Any

class ToolManager:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.gemini_tools: List[Callable] = []

    def register_tool(self, func: Callable):
        """Registers a Python function as a tool for the agent."""
        self.tools[func.__name__] = func
        self.gemini_tools.append(func)
        return func

    def get_tools_for_gemini(self):
        """Returns the list of tools formatted for Gemini."""
        # Gemini Python SDK handles function conversion automatically when passed to tools
        return self.gemini_tools if self.gemini_tools else None
    
    def get_tool(self, tool_name: str) -> Callable:
        """Returns a tool function by name."""
        return self.tools.get(tool_name)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]):
        """Executes a tool call locally (if needed for manual handling)."""
        if tool_name in self.tools:
            return self.tools[tool_name](**args)
        else:
            raise ValueError(f"Tool {tool_name} not found.")
