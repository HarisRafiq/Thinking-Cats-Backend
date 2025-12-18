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
                        system_instruction: Optional[str] = None,
                        generation_config: Optional[Dict[str, Any]] = None) -> Any:
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
        
    def _get_model(self, tools: Optional[List[Callable]] = None, system_instruction: Optional[str] = None, generation_config: Optional[Dict[str, Any]] = None):
        """Lazily initializes or updates the model if configuration changes."""
        # In Gemini, tools and system_instruction are set at model initialization
        # If they change, we need to re-initialize the model
        if (self.model is None or 
            tools != self._current_tools or 
            system_instruction != self._current_system_instruction or
            generation_config != getattr(self, '_current_generation_config', None)):
            
            self._current_tools = tools
            self._current_system_instruction = system_instruction
            self._current_generation_config = generation_config
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                tools=tools,
                system_instruction=system_instruction,
                generation_config=generation_config
            )
        return self.model

    def generate_content(self, 
                        prompt: Union[str, List[Any]], 
                        tools: Optional[List[Callable]] = None,
                        system_instruction: Optional[str] = None,
                        generation_config: Optional[Dict[str, Any]] = None) -> Any:
        
        model = self._get_model(tools, system_instruction, generation_config)
        return model.generate_content(prompt)

    async def generate_content_async(self, 
                        prompt: Union[str, List[Any]], 
                        tools: Optional[List[Callable]] = None,
                        system_instruction: Optional[str] = None,
                        generation_config: Optional[Dict[str, Any]] = None) -> Any:
        
        model = self._get_model(tools, system_instruction, generation_config)
        return await model.generate_content_async(prompt)
    
    async def generate(
        self,
        prompt: Union[str, List[Any]],
        system_instruction: Optional[str] = None
    ) -> str:
        """Simple async generate that returns just the text."""
        model = self._get_model(tools=None, system_instruction=system_instruction)
        response = await model.generate_content_async(prompt)
        return response.text if response.text else ""
    
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

    async def generate_stream(
        self,
        prompt: Union[str, List[Any]],
        system_instruction: Optional[str] = None
    ):
        """
        Generates content with streaming for real-time responses.
        Yields chunks as they arrive.
        """
        model = self._get_model(tools=None, system_instruction=system_instruction)
        
        response = await model.generate_content_async(
            prompt,
            stream=True
        )
        
        async for chunk in response:
            if chunk.text:
                yield {"text": chunk.text}

