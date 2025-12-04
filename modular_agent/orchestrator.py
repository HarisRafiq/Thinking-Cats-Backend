from typing import List, Dict, Optional, Callable, Any
import google.generativeai as genai
import asyncio
from .core import ModularAgent
from .personalities import PersonalityManager
from .llm import GeminiProvider
from .database import DatabaseManager
from .tools.definitions import ToolDefinitions
from .utils.sanitization import sanitize_response

class Orchestrator:
    def __init__(self, model_name: str = "gemini-2.5-flash", verbose: bool = True, event_callback: Optional[Callable[[Dict], None]] = None, theme: str = "cat", session_id: Optional[str] = None, user_id: Optional[str] = None, db_manager: Optional[DatabaseManager] = None):
        self.verbose = verbose
        self.model_name = model_name
        self.personality_manager = PersonalityManager(db_manager=db_manager)
        self.event_callback = event_callback
        self.theme = theme
        # waiting_for_clarification removed - now using DB state
        self.active_experts = {} # Map real_name -> fictional_name
        self.pending_questions = {} # Map fictional_name -> sanitized_question
        self.session_id = session_id
        self.user_id = user_id
        self.db_manager = db_manager
        
        # Initialize provider
        self.provider = GeminiProvider(model_name=model_name)
        
        # Create a simple model instance for generating one-liners
        # We can use the provider for this now
        self._one_liner_provider = GeminiProvider(model_name="gemini-2.5-flash-lite")
        
        # Initialize tool definitions
        self.tool_definitions = ToolDefinitions(self)

        # Initialize the main orchestrator agent
        self.agent = ModularAgent(
            provider=self.provider,
            personality="moderator",
            tools=[self.tool_definitions.consult_expert, self.tool_definitions.ask_clarification],
            verbose=verbose
        )





    def _log_usage_background(self, usage: Dict[str, int], model: str = "gemini-2.5-flash", prompt: str = None, response: str = None):
        """Updates user usage and logs LLM call in background."""
        if not self.db_manager or not self.user_id:
            return
            
        async def _update():
            try:
                # Cost calculation (approximate for Gemini Flash)
                # Input: $0.075 / 1M tokens
                # Output: $0.30 / 1M tokens
                # Thinking tokens are priced as output tokens (usually)
                thinking_tokens = usage.get('thinking_tokens', 0)
                output_tokens = usage['output_tokens']
                
                # Total output-equivalent tokens for cost
                total_output_equivalent = output_tokens + thinking_tokens
                
                cost = (usage['input_tokens'] * 0.075 / 1_000_000) + (total_output_equivalent * 0.30 / 1_000_000)
                
                # Update user usage stats
                await self.db_manager.update_user_usage(
                    self.user_id, 
                    usage['input_tokens'], 
                    output_tokens, 
                    thinking_tokens,
                    cost
                )
                
                # Log detailed LLM call
                await self.db_manager.log_llm_call({
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "model": model,
                    "input_tokens": usage['input_tokens'],
                    "output_tokens": output_tokens,
                    "thinking_tokens": thinking_tokens,
                    "total_tokens": usage['total_tokens'],
                    "cost": cost,
                    "prompt": prompt[:2000] if prompt else None, # Truncate for storage
                    "response": response[:2000] if response else None # Truncate for storage
                })
                
            except Exception as e:
                if self.verbose:
                    print(f"[Orchestrator] Error updating usage/logging: {e}")
        
        asyncio.create_task(_update())

    async def _create_user_message_slide(self, content: str, is_clarification_response: bool = False, pending_interaction: Optional[Dict[str, Any]] = None):
        """Helper method to create and store a user_message slide.
        
        If is_clarification_response is True, this creates a Q&A format user_message slide
        directly using the question from pending_interaction.
        """
        if not self.db_manager or not self.session_id:
            return
        
        # Check if this is a response to a clarification
        if is_clarification_response:
            # Get the question from pending_interaction
            question = pending_interaction.get("question", "") if pending_interaction else ""
            
            if question:
                # Create a new Q&A user_message slide directly
                try:
                    import uuid
                    temp_id = str(uuid.uuid4())
                    
                    slide = {
                        "type": "user_message",
                        "question": question,
                        "answer": content,
                        "id": temp_id
                    }
                    
                    # Create slide in DB synchronously
                    slide_id = await self.db_manager.add_slide(self.session_id, slide)
                    slide["id"] = slide_id  # Update with the real MongoDB ObjectId
                    
                    # Emit slide_added event with the real MongoDB ID
                    await self._emit_event({
                        "type": "slide_added",
                        "slide": slide
                    })
                    
                    if self.verbose:
                        print(f"[Orchestrator] Created Q&A user_message slide for clarification response")
                except Exception as e:
                    if self.verbose:
                        print(f"[Orchestrator] Error creating Q&A slide for clarification response: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Emit event to notify frontend
                await self._emit_event({
                    "type": "clarification_answered",
                    "selectedOption": content
                })
            
            # Continue to let the agent process the answer
        else:
            # Generate temporary ID first for immediate event emission
            import uuid
            temp_id = str(uuid.uuid4())
            
            slide = {
                "type": "user_message",
                "content": content,
                "id": temp_id  # Temporary ID, will be updated by background task
            }
            
            # Create slide in DB in background (non-blocking)
            async def _create_slide_background():
                try:
                    slide_id = await self.db_manager.add_slide(self.session_id, slide)
                    # Update slide ID if different from temp (though usually same)
                    if slide_id != temp_id:
                        slide["id"] = slide_id
                except Exception as e:
                    if self.verbose:
                        print(f"[Orchestrator] Error creating slide in DB: {e}")
                    # Keep temporary ID if DB write fails
            
            # Start background task for slide creation (fire-and-forget)
            asyncio.create_task(_create_slide_background())
            
            # Emit slide_added event immediately (non-blocking)
            await self._emit_event({
                "type": "slide_added",
                "slide": slide
            })

    async def _emit_event(self, event_data: Dict):
        """Helper method to emit events, handling null callback. Non-blocking DB writes."""
        # Add timestamp if not present
        if "timestamp" not in event_data:
            import time
            event_data["timestamp"] = time.time()

        # Emit event immediately to callback (non-blocking)
        if self.event_callback:
            # Check if callback is async
            if asyncio.iscoroutinefunction(self.event_callback):
                await self.event_callback(event_data)
            else:
                self.event_callback(event_data)
        
        # Save event to DB in background task (fire-and-forget pattern)
        # This prevents blocking the event emission path
        if self.db_manager and self.session_id:
            async def _save_event_background():
                try:
                    await self.db_manager.add_event(self.session_id, event_data)
                except Exception as e:
                    # Log error but don't block event emission
                    if self.verbose:
                        print(f"[Orchestrator] Error saving event to DB (non-critical): {e}")
            
            # Create background task for DB write
            asyncio.create_task(_save_event_background())

    async def process(self, user_input: str) -> str:
        """
        Processes user input by deciding whether to ask for clarification or summon experts.
        The orchestrator does NOT generate text responses - it only uses tools.
        """
        if self.verbose:
            print(f"\n[Orchestrator] Received input: {user_input}")
            
        # Check current session status
        is_clarification_response = False
        pending_interaction = None
        if self.db_manager and self.session_id:
            session = await self.db_manager.get_session(self.session_id)
            if session and session.get("status") == "waiting_for_input":
                pending = session.get("pending_interaction", {})
                if pending and pending.get("type") == "clarification":
                    is_clarification_response = True
                    pending_interaction = pending  # Save this BEFORE we clear it
                    if self.verbose:
                        print(f"[Orchestrator] Handling input as clarification response")
        
        # Update status to processing
        # Explicitly clear pending_interaction when processing a response (user has answered)
        if self.db_manager and self.session_id:
            await self.db_manager.update_session_status(self.session_id, "processing", None)

        # Create user_message slide for the input
        # If it's a clarification response, this method handles the conversion
        await self._create_user_message_slide(user_input, is_clarification_response=is_clarification_response, pending_interaction=pending_interaction)


        # Emit thinking event
        await self._emit_event({
            "type": "orchestrator_thinking",
            "label": "Thinking Cats are planning..."
        })
        
        # Construct prompt that instructs orchestrator to only use tools, never generate text
        prompt = (
            f"User input: {user_input}\n\n"
            "Your job is to create a compelling report on the given user prompt. You have the context history of the conversation for reference. You have the ability to summon any famous personality from history or real world to help you finish up the report."
        )
        response, usage = await self.agent.chat(prompt)
        self._log_usage_background(usage, model=self.model_name, prompt=prompt, response=response)
        
        # If we reached here without raising AgentInterruption (i.e. no clarification asked),
        # we are done with this turn. Set status to idle.
        if self.db_manager and self.session_id:
             # Check if we are actually waiting for input (AgentInterruption would have been raised, but just in case)
             # Only set to idle if we're not waiting for input
             session = await self.db_manager.get_session(self.session_id)
             if session and session.get("status") != "waiting_for_input":
                 # If the agent just finished normally and we're not waiting for input, we are idle.
                 await self.db_manager.update_session_status(self.session_id, "idle")

        
        # The orchestrator should not generate text responses, but if it does (fallback),
        # we don't emit events or create slides for it since it's not supposed to happen.
        # The agent's tool calls will have already emitted their events.
        
        return response

    async def load_session_history(self):
        """Loads session history from the database and hydrates the agent context."""
        if not self.db_manager or not self.session_id:
            return
            
        session = await self.db_manager.get_session(self.session_id)
        if not session:
            return
            
        print(f"[Orchestrator] Loading history for session {self.session_id}")
        
        
        # Combine messages and events into a single timeline
        timeline = []
        
        # Add messages
        if 'messages' in session:
            for msg in session['messages']:
                timeline.append({
                    'type': 'message',
                    'data': msg,
                    'timestamp': msg.get('timestamp', 0)
                })
                
        # Add events
        if 'events' in session:
            for event in session['events']:
                timeline.append({
                    'type': 'event',
                    'data': event,
                    'timestamp': event.get('timestamp', 0)
                })
                
        # Sort by timestamp
        timeline.sort(key=lambda x: x['timestamp'])
        
        # Reconstruct context
        self.agent.context.clear()
        
        # We don't need pending_clarification_response flag anymore for logic flow, 
        # but we might need it to reconstruct the conversation correctly if we want to 
        # simulate the function response.
        # Actually, if the session status is 'waiting_for_input', the last item should be a clarification request.
        # If we are loading history, we just replay what happened.
        
        pending_clarification_response = False

        
        for item in timeline:
            if item['type'] == 'message':
                msg = item['data']
                if msg.get('role') == 'user':
                    content = msg.get('content')
                    
                    # If we're waiting for a clarification response, add it as function response
                    # Use just the answer (question already in context from clarification_request)
                    if pending_clarification_response:
                        from google.generativeai import protos
                        
                        # Check if content is in old "Q: ... A: ..." format and extract answer
                        answer_to_use = content
                        if content.startswith("Q:") and "\nA:" in content:
                            # Old format - extract just the answer for token efficiency
                            parts = content.split("\nA:", 1)
                            if len(parts) == 2:
                                answer_to_use = parts[1].strip()
                        # If there's a Q&A slide with this answer, we could use it, but content should work
                        # (Q&A slides are for display, messages are for context)
                        
                        fn_response = protos.FunctionResponse(
                            name='ask_clarification',
                            response={'result': answer_to_use}
                        )
                        self.agent.context.add_function_response(fn_response)
                        pending_clarification_response = False
                    else:
                        # For non-Q&A messages, use content as-is
                        self.agent.context.add_user_message(content)
            
            elif item['type'] == 'event':
                event = item['data']
                event_type = event.get('type')
                
                if event_type == 'consult_start':
                    # This corresponds to a function call by the model
                    real_expert = event.get('expert')  # Real name (e.g., "Steve Jobs")
                    fictional_name = event.get('fictional_name')  # Fictional name (e.g., "Steve Paws")
                    question = event.get('question')
                    
                    # Restore the active expert mapping (real -> fictional)
                    if real_expert and fictional_name:
                        self.active_experts[real_expert] = fictional_name
                    
                    # Add model message with function call using REAL name
                    # (The orchestrator uses real names internally)
                    from google.generativeai import protos
                    fn_call = protos.FunctionCall(
                        name='consult_expert',
                        args={'expert_name': real_expert, 'question': question}
                    )
                    self.agent.context.add_model_message("", function_call=fn_call)
                    
                elif event_type == 'consult_end':
                    # This corresponds to a function response
                    real_expert = event.get('expert')  # Real name
                    response = event.get('response')
                    
                    # Function response uses REAL name (backend logic)
                    from google.generativeai import protos
                    fn_response = protos.FunctionResponse(
                        name='consult_expert',
                        response={'result': f"[{real_expert}]: {response}"}
                    )
                    self.agent.context.add_function_response(fn_response)
                    
                elif event_type == 'clarification_request':
                    # This corresponds to an ask_clarification function call
                    question = event.get('question')
                    options = event.get('options', [])
                    
                    from google.generativeai import protos
                    fn_call = protos.FunctionCall(
                        name='ask_clarification',
                        args={'question': question, 'options': options}
                    )
                    self.agent.context.add_model_message("", function_call=fn_call)
                    
                    # The next user message after this is the response to the clarification
                    pending_clarification_response = True

        print(f"[Orchestrator] History loaded. Context length: {len(self.agent.context)}")
