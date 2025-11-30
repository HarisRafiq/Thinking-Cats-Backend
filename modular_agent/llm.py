from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Union
import google.generativeai as genai
from google.generativeai import protos

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_content_async(self, 
                        prompt: Union[str, List[Any]], 
                        tools: Optional[List[Callable]] = None,
                        system_instruction: Optional[str] = None) -> Any:
        """Generates content from the model asynchronously."""
        pass
    
    @abstractmethod
    def get_token_usage(self, response: Any) -> Dict[str, int]:
        """Extracts token usage from the response."""
        pass

class GeminiProvider(LLMProvider):
    """Gemini implementation of LLMProvider."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        if api_key:
            genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = None
        self._current_tools = None
        self._current_system_instruction = None
        
    def _get_model(self, tools: Optional[List[Callable]] = None, system_instruction: Optional[str] = None):
        """Lazily initializes or updates the model if configuration changes."""
        # In Gemini, tools and system_instruction are set at model initialization
        # If they change, we need to re-initialize the model
        if self.model is None or tools != self._current_tools or system_instruction != self._current_system_instruction:
            self._current_tools = tools
            self._current_system_instruction = system_instruction
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                tools=tools,
                system_instruction=system_instruction
            )
        return self.model

    def generate_content(self, 
                        prompt: Union[str, List[Any]], 
                        tools: Optional[List[Callable]] = None,
                        system_instruction: Optional[str] = None) -> Any:
        
        model = self._get_model(tools, system_instruction)
        return model.generate_content(prompt)

    async def generate_content_async(self, 
                        prompt: Union[str, List[Any]], 
                        tools: Optional[List[Callable]] = None,
                        system_instruction: Optional[str] = None) -> Any:
        
        model = self._get_model(tools, system_instruction)
        return await model.generate_content_async(prompt)
    
    def get_token_usage(self, response: Any) -> Dict[str, int]:
        usage = {
            'input_tokens': 0,
            'output_tokens': 0,
            'thinking_tokens': 0,
            'total_tokens': 0
        }
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage['input_tokens'] = getattr(response.usage_metadata, 'prompt_token_count', 0)
            usage['output_tokens'] = getattr(response.usage_metadata, 'candidates_token_count', 0)
            usage['total_tokens'] = getattr(response.usage_metadata, 'total_token_count', 0)
            
            # Calculate thinking tokens (Total - (Input + Output))
            # This handles models like gemini-2.0-flash-thinking-exp where thinking tokens are included in total but not separated
            usage['thinking_tokens'] = max(0, usage['total_tokens'] - (usage['input_tokens'] + usage['output_tokens']))
        return usage
