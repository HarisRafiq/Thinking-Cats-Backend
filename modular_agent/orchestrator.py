from typing import List, Dict, Optional, Callable
import google.generativeai as genai
import asyncio
from .core import ModularAgent
from .personalities import PersonalityManager
from .llm import GeminiProvider
from .database import DatabaseManager

class Orchestrator:
    def __init__(self, model_name: str = "gemini-2.5-flash", verbose: bool = True, event_callback: Optional[Callable[[Dict], None]] = None, theme: str = "cat", session_id: Optional[str] = None, db_manager: Optional[DatabaseManager] = None):
        self.verbose = verbose
        self.model_name = model_name
        self.personality_manager = PersonalityManager()
        self.event_callback = event_callback
        self.theme = theme
        self.waiting_for_clarification = False
        self.active_experts = {} # Map real_name -> fictional_name
        self.pending_questions = {} # Map fictional_name -> sanitized_question
        self.session_id = session_id
        self.db_manager = db_manager
        
        # Initialize provider
        self.provider = GeminiProvider(model_name=model_name)
        
        # Create a simple model instance for generating one-liners
        # We can use the provider for this now
        self._one_liner_provider = GeminiProvider(model_name=model_name)
        
        # Define the tool for the orchestrator
        async def consult_expert(expert_name: str, question: str) -> str:
            """
            Consults a famous personality or expert with a question.
            
            Args:
                expert_name: The name of the famous person or expert to consult (e.g., 'Steve Jobs', 'Marie Curie', 'Sherlock Holmes').
                question: The specific question or context to provide to the expert.
                
            Returns:
                The expert's response.
            """
            if self.verbose:
                print(f"\n[Orchestrator] Consulting {expert_name}...")
            
            # Initialize fictional_name for error handling
            fictional_name = expert_name
            
            try:
                # Check if it's a standard personality, otherwise create a dynamic one
                if expert_name.lower() in self.personality_manager.personalities:
                    if expert_name.lower() == 'synthesizer':
                         personality = self.personality_manager.get_personality('synthesizer')
                         agent = ModularAgent(provider=self.provider, personality='synthesizer', verbose=False)
                         one_liner = personality.one_liner or f"Expert: {expert_name}"
                         fictional_name = personality.fictional_name or expert_name
                    else:
                        # For standard personalities, we still might want to use the dynamic creation logic 
                        # if we want to ensure consistency, but let's stick to the existing logic
                        # actually the existing logic for standard personalities was a bit mixed.
                        # Let's simplify: if it's in manager, use it.
                        personality = self.personality_manager.get_personality(expert_name.lower())
                        agent = ModularAgent(provider=self.provider, personality=expert_name.lower(), verbose=False)
                        one_liner = personality.one_liner or f"Famous figure: {expert_name}"
                        fictional_name = personality.fictional_name or expert_name
                else:
                    # Dynamic personality
                    # We need to pass a model to create_dynamic_personality, but it expects a genai.GenerativeModel
                    # Let's expose the model from provider or just create one temporarily
                    # Ideally create_dynamic_personality should be updated, but for now let's use the provider's internal model if possible
                    # or just pass the provider and update create_dynamic_personality.
                    # For now, let's just create a genai model as before for this specific helper
                    _temp_model = genai.GenerativeModel(model_name=self.model_name)
                    personality = self.personality_manager.create_dynamic_personality(expert_name, _temp_model, theme=self.theme)
                    
                    agent = ModularAgent(provider=self.provider, personality='default', verbose=False)
                    agent.current_personality = personality
                    # agent._setup_model() # No longer needed as provider handles it
                    one_liner = personality.one_liner or f"Famous figure: {expert_name}"
                    fictional_name = personality.fictional_name or expert_name
                
                # Store mapping for global sanitization
                self.active_experts[expert_name] = fictional_name
                
                # Sanitize question for the event (so frontend doesn't see real names in the question text)
                sanitized_question = self._sanitize_response(question, expert_name, fictional_name)
                
                # Store sanitized question for later use in slide creation
                self.pending_questions[fictional_name] = sanitized_question
                
                # Send consult_start event with real name and fictional name
                # Backend uses 'expert' (real name), frontend uses 'fictional_name'
                await self._emit_event({
                    "type": "consult_start",
                    "expert": expert_name,  # Real name for backend
                    "fictional_name": fictional_name,  # Display name for frontend
                    "question": sanitized_question,
                    "one_liner": one_liner
                })

                response = await agent.run(question)
                
                if self.verbose:
                    print(f"[{expert_name}] {response[:100]}..." if len(response) > 100 else f"[{expert_name}] {response}")
                
                # Sanitize response before sending to frontend
                sanitized_response = self._sanitize_response(response, expert_name, fictional_name)
                
                await self._emit_event({
                    "type": "consult_end",
                    "expert": expert_name,  # Real name for backend
                    "fictional_name": fictional_name,  # Display name for frontend
                    "response": sanitized_response,
                    "one_liner": one_liner  # Include for consistency
                })
                
                # Create agent_response slide
                if self.db_manager and self.session_id:
                    slide = {
                        "type": "agent_response",
                        "sender": fictional_name,
                        "content": sanitized_response,
                        "question": self.pending_questions.get(fictional_name, ""),
                        "oneLiner": one_liner
                    }
                    slide_id = await self.db_manager.add_slide(self.session_id, slide)
                    slide["id"] = slide_id
                    
                    # Emit slide_added event
                    await self._emit_event({
                        "type": "slide_added",
                        "slide": slide
                    })
                    
                # Return the REAL name to the orchestrator so it knows who it talked to
                return f"[{expert_name}]: {response}"
            except Exception as e:
                error_msg = f"Error consulting {expert_name}: {str(e)}"
                await self._emit_event({
                    "type": "error",
                    "expert": expert_name,
                    "message": error_msg
                })
                return error_msg

        async def ask_clarification(question: str) -> str:
            """
            Asks the user a clarifying question when the initial problem statement is insufficient.
            
            Args:
                question: The clarifying question to ask the user.
                
            Returns:
                The user's answer to the question.
            """
            if self.verbose:
                print(f"\n[Orchestrator] Asking for clarification: {question}")
            
            self.waiting_for_clarification = True
            
            await self._emit_event({
                "type": "clarification_request",
                "question": question
            })
            
            # In a real interactive loop, we would wait for input here.
            # For now, we return a placeholder or stop execution depending on the runner.
            # The frontend will handle the event and prompt the user.
            return "WAITING_FOR_USER_INPUT"

        # Initialize the main orchestrator agent
        self.agent = ModularAgent(
            provider=self.provider,
            personality="moderator",
            tools=[consult_expert, ask_clarification],
            verbose=verbose
        )

    def _sanitize_response(self, text: str, real_name: str = None, fictional_name: str = None) -> str:
        """
        Replaces real names with fictional names in the text.
        If real_name/fictional_name are provided, it prioritizes them.
        It also checks self.active_experts for other replacements.
        """
        sanitized = text
        
        # Helper to replace variants
        def replace_variants(content, real, fictional):
            if not real or not fictional or real == fictional:
                return content
            
            # Replace full name
            content = content.replace(real, fictional)
            
            # Replace last name (simple heuristic)
            real_parts = real.split()
            fictional_parts = fictional.split()
            if len(real_parts) > 1 and len(fictional_parts) > 1:
                content = content.replace(real_parts[-1], fictional_parts[-1])

            # Replace first name as well
            if len(real_parts) > 0 and len(fictional_parts) > 0:
                content = content.replace(real_parts[0], fictional_parts[0])
                
            return content

        # 1. Prioritize the current expert if provided
        if real_name and fictional_name:
            sanitized = replace_variants(sanitized, real_name, fictional_name)
            
        # 2. Replace all other known experts
        for r_name, f_name in self.active_experts.items():
            if r_name != real_name: # Already handled
                sanitized = replace_variants(sanitized, r_name, f_name)
                
        return sanitized

    async def _create_user_message_slide(self, content: str):
        """Helper method to create and store a user_message slide."""
        if not self.db_manager or not self.session_id:
            return
        
        slide = {
            "type": "user_message",
            "content": content
        }
        slide_id = await self.db_manager.add_slide(self.session_id, slide)
        slide["id"] = slide_id
        
        # Emit slide_added event
        await self._emit_event({
            "type": "slide_added",
            "slide": slide
        })

    async def _emit_event(self, event_data: Dict):
        """Helper method to emit events, handling null callback."""
        # Add timestamp if not present
        if "timestamp" not in event_data:
            import time
            event_data["timestamp"] = time.time()

        if self.event_callback:
            # Check if callback is async
            if asyncio.iscoroutinefunction(self.event_callback):
                await self.event_callback(event_data)
            else:
                self.event_callback(event_data)
        
        # Save event to DB if configured
        if self.db_manager and self.session_id:
            await self.db_manager.add_event(self.session_id, event_data)

    async def process(self, user_input: str) -> str:
        """
        Processes user input by deciding whether to ask for clarification or summon experts.
        The orchestrator does NOT generate text responses - it only uses tools.
        """
        if self.verbose:
            print(f"\n[Orchestrator] Received input: {user_input}")
        
        # Create user_message slide for the input
        await self._create_user_message_slide(user_input)
        
        # Construct prompt that instructs orchestrator to only use tools, never generate text
        prompt = (
            f"User input: {user_input}\n\n"
            "You are the moderator of a roundtable discussion. You have the ability to summon any famous personality from history or real world that are best suited to help solve the problem.\n"
            "Your goal is to coordinate a diverse team of these famous figures that each work progressively to reach the solution.\n\n"
            "CRITICAL RULES:\n"
            "1. You MUST ONLY use tools. NEVER generate text responses or explanations.\n"
            "2. ALWAYS use the 'consult_expert' tool to summon a famous person. Do not simulate their responses.\n"
            "3. Call ONE person at a time. And Only ask ONE question at a time that is missing from the conversation history and is relevant to the problem.\n"
            "4. Choose personalities that can offer the best advice on the missing piece of the puzzle.\n"               
            "5. If the input is VAGUE or lacks sufficient detail, use the 'ask_clarification' tool to ask the user for more information BEFORE consulting any experts. But don't be too strict about it. Automatically assume where necessary.\n"
            "6. You can iterate through the process as many times as needed.\n"
            "7. Start by analyzing the input. If it's clear, summon the most relevant famous figure. If it's vague, ask for clarification.\n"
            "8. After roundtable discussion is complete, just stop.\n"
            "9. Do not ask the same question to the same person more than once.\n"
            "10. REMEMBER: You are a coordinator only. Use tools, do not speak."
        )
        
        self.waiting_for_clarification = False
        response = await self.agent.chat(prompt)
        
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
        
        for item in timeline:
            if item['type'] == 'message':
                msg = item['data']
                if msg.get('role') == 'user':
                    self.agent.context.add_user_message(msg.get('content'))
            
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

        print(f"[Orchestrator] History loaded. Context length: {len(self.agent.context)}")
