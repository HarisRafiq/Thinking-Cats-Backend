import google.generativeai as genai
from typing import List, Optional, Union, Callable, Dict, Any
from .config import DEFAULT_MODEL
from .personalities import PersonalityManager, Personality
from .tools import ToolManager
from .context_manager import ContextManager
from .llm import LLMProvider, GeminiProvider

class AgentInterruption(Exception):
    """Raised by tools to stop agent execution immediately."""
    pass

class ModularAgent:
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        model_name: str = DEFAULT_MODEL, # Kept for backward compatibility/default provider
        personality: str = "default",
        tools: Optional[List[Callable]] = None,
        verbose: bool = False
    ):
        self.verbose = verbose
        
        # Initialize Provider
        if provider:
            self.provider = provider
        else:
            self.provider = GeminiProvider(model_name=model_name)
        
        # Initialize Personality
        self.personality_manager = PersonalityManager()
        self.current_personality = self.personality_manager.get_personality(personality)
        
        # Initialize Tools
        self.tool_manager = ToolManager()
        if tools:
            for tool in tools:
                self.tool_manager.register_tool(tool)
        
        # Context Management
        self.context = ContextManager(verbose=verbose)
        
        # Token Usage Tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_thinking_tokens = 0
        self.total_tokens = 0

    def set_personality(self, personality_name: str):
        """Switches the agent's personality."""
        self.current_personality = self.personality_manager.get_personality(personality_name)
        # Reset context as context might be invalid for new personality
        self.context.clear()
    
    def get_token_usage(self) -> dict:
        """Returns current token usage statistics."""
        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'thinking_tokens': self.total_thinking_tokens,
            'total_tokens': self.total_tokens
        }
    
    def reset_token_usage(self):
        """Resets token usage counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_thinking_tokens = 0
        self.total_tokens = 0

    async def chat(self, user_input: str) -> str:
        """
        Main conversational method using generate_content_async() with manual context management.
        """
        # Add current user message to context
        self.context.add_user_message(user_input)
        
        # Capture start tokens for delta calculation
        start_input_tokens = self.total_input_tokens
        start_output_tokens = self.total_output_tokens
        start_thinking_tokens = self.total_thinking_tokens
        start_total_tokens = self.total_tokens
        
        # Manual function calling loop to ensure we get a final text response
        max_iterations = 20  # Prevent infinite loops
        iteration = 0
        final_response_text = ""
        
        try:
            # Build context for initial API call
            contents = self.context.build_api_context(include_current_user=False)
            
            # Initial API call
            system_instruction = self.current_personality.system_instruction
            tools = self.tool_manager.get_tools_for_gemini()
            
            response = await self.provider.generate_content_async(
                prompt=contents,
                tools=tools,
                system_instruction=system_instruction
            )
            
            # Manual function calling loop
            while iteration < max_iterations:
                # Track token usage for this iteration
                usage = self.provider.get_token_usage(response)
                self.total_input_tokens += usage['input_tokens']
                self.total_output_tokens += usage['output_tokens']
                self.total_thinking_tokens += usage.get('thinking_tokens', 0)
                self.total_tokens += usage['total_tokens']
                
                if self.verbose:
                    print(f"\n[Token Usage] Input: {usage['input_tokens']}, Output: {usage['output_tokens']}, Total: {usage['total_tokens']}")
                    print(f"[Cumulative Tokens] Input: {self.total_input_tokens}, Output: {self.total_output_tokens}, Total: {self.total_tokens}")
                
                # Check if there are function calls to execute
                function_calls = []
                response_text = ""
                
                if response.candidates and len(response.candidates) > 0:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls.append(part.function_call)
                        elif hasattr(part, 'text') and part.text:
                            response_text += part.text
                
                # If no function calls, we have a final text response
                if not function_calls:
                    final_response_text = response_text
                    # Add final model response to context
                    if final_response_text:
                        self.context.add_model_message(final_response_text)
                    break
                
                if self.verbose:
                    for fn in function_calls:
                        print(f"\n[Thinking] I need to use the tool '{fn.name}' with arguments: {dict(fn.args)}")
                
                # Add model message with function calls to context
                for fn in function_calls:
                    self.context.add_model_message("", function_call=fn)
                
                # Execute function calls and prepare responses
                for fn in function_calls:
                    try:
                        # Get the tool function from the tool manager
                        tool_func = self.tool_manager.get_tool(fn.name)
                        if tool_func is None:
                            raise ValueError(f"Tool '{fn.name}' not found")
                        
                        # Execute the tool
                        # Check if tool is async
                        import inspect
                        if inspect.iscoroutinefunction(tool_func):
                            result = await tool_func(**dict(fn.args))
                        else:
                            result = tool_func(**dict(fn.args))
                        
                        if self.verbose:
                            print(f"[Tool Output] {fn.name}: {result}")
                        
                        # Create function response
                        from google.generativeai import protos
                        fn_response = protos.FunctionResponse(name=fn.name, response={"result": result})
                        if hasattr(fn, 'id') and fn.id:
                            fn_response.id = fn.id
                        
                        # Add function response to context (Gemini expects function responses as user role)
                        self.context.add_function_response(fn_response)
                        
                    except AgentInterruption as e:
                        if self.verbose:
                            print(f"[System] Agent execution interrupted: {str(e)}")
                        # We stop the loop immediately. 
                        # We do NOT add a function response because we want the conversation to effectively pause/end here.
                        # Or maybe we should add it so the model knows what happened if we resume?
                        # For clarification, the user will reply, and we will likely start a NEW turn or resume.
                        # If we resume, we need the history.
                        # Let's add the interruption message as a result so history is consistent.
                        from google.generativeai import protos
                        fn_response = protos.FunctionResponse(name=fn.name, response={"result": str(e)})
                        if hasattr(fn, 'id') and fn.id:
                            fn_response.id = fn.id
                        self.context.add_function_response(fn_response)
                        
                        # Calculate usage for this chat turn before returning
                        current_usage = {
                            'input_tokens': self.total_input_tokens - start_input_tokens,
                            'output_tokens': self.total_output_tokens - start_output_tokens,
                            'thinking_tokens': self.total_thinking_tokens - start_thinking_tokens,
                            'total_tokens': self.total_tokens - start_total_tokens
                        }
                        return str(e), current_usage

                    except Exception as e:
                        if self.verbose:
                            print(f"[Tool Error] Error executing {fn.name}: {str(e)}")
                        from google.generativeai import protos
                        fn_response = protos.FunctionResponse(name=fn.name, response={"error": str(e)})
                        if hasattr(fn, 'id') and fn.id:
                            fn_response.id = fn.id
                        
                        # Add error response to context
                        self.context.add_function_response(fn_response)
                
                # Build context for next iteration (includes function calls and responses we just added)
                contents = self.context.build_api_context(include_current_user=False)
                
                # Make next API call
                response = await self.provider.generate_content_async(
                    prompt=contents,
                    tools=tools,
                    system_instruction=system_instruction
                )
                iteration += 1
            
            
            if iteration >= max_iterations:
                print(f"\n[Warning] Reached maximum function calling iterations ({max_iterations})")
                if not final_response_text and response_text:
                    final_response_text = response_text
                    self.context.add_model_message(final_response_text)
            
        except Exception as e:
            # Remove the user message we added if there was an error
            if len(self.context) > 0:
                last_msg = self.context.messages[-1]
                if last_msg.get("role") == "user" and last_msg.get("content") == user_input:
                    self.context.messages.pop()
            
            if "MALFORMED_FUNCTION_CALL" in str(e):
                # Retry once with a reminder
                print("\n[System] Malformed function call detected. Retrying with reminder...")
                # Add reminder as user message
                self.context.add_user_message("You made a malformed function call. Please ensure you are calling the tool correctly with valid JSON arguments.")
                contents = self.context.build_api_context(include_current_user=False)
                response = await self.provider.generate_content_async(
                    prompt=contents,
                    tools=tools,
                    system_instruction=system_instruction
                )
                if response.candidates and len(response.candidates) > 0:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            final_response_text = part.text
                            self.context.add_model_message(final_response_text)
            else:
                raise e
        
        # Calculate usage for this chat turn
        current_usage = {
            'input_tokens': self.total_input_tokens - start_input_tokens,
            'output_tokens': self.total_output_tokens - start_output_tokens,
            'thinking_tokens': self.total_thinking_tokens - start_thinking_tokens,
            'total_tokens': self.total_tokens - start_total_tokens
        }
        
        return final_response_text, current_usage

    async def run(self, user_input: str) -> str:
        """Single-shot execution (no history)."""
        system_instruction = self.current_personality.system_instruction
        tools = self.tool_manager.get_tools_for_gemini()
        
        response = await self.provider.generate_content_async(
            prompt=user_input,
            tools=tools,
            system_instruction=system_instruction
        )
        
        # Track token usage
        usage = self.provider.get_token_usage(response)
        self.total_input_tokens += usage['input_tokens']
        self.total_output_tokens += usage['output_tokens']
        self.total_thinking_tokens += usage.get('thinking_tokens', 0)
        self.total_tokens += usage['total_tokens']
        
        if self.verbose:
            print(f"\n[Token Usage] Input: {usage['input_tokens']}, Output: {usage['output_tokens']}, Total: {usage['total_tokens']}")
            print(f"[Cumulative Tokens] Input: {self.total_input_tokens}, Output: {self.total_output_tokens}, Total: {self.total_tokens}")
        
        return response.text, usage

    def as_tool(self) -> Callable:
        """
        Returns a callable that can be used as a tool by another agent.
        This function wraps the agent's 'run' method.
        """
        def agent_tool(query: str):
            """Consults the sub-agent to answer a query."""
            response, _ = self.run(query)
            return response
        
        agent_tool.__name__ = f"ask_{self.current_personality.name.lower()}_agent"
        return agent_tool
