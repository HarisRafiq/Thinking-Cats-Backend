from typing import List, Dict, Optional, Callable, Any, TYPE_CHECKING
import asyncio
from ..core import ModularAgent, AgentInterruption
from ..utils.sanitization import sanitize_response

if TYPE_CHECKING:
    from ..orchestrator import Orchestrator

class ToolDefinitions:
    def __init__(self, orchestrator: 'Orchestrator'):
        self.orchestrator = orchestrator

    async def consult_expert(self, expert_name: str, question: str) -> str:
        """
        Consults a famous personality or expert with a question.
        
        Args:
            expert_name: The name of the famous person or expert to consult (e.g., 'Steve Jobs', 'Marie Curie', 'Sherlock Holmes').
            question: The specific question or context to provide to the expert.
            
        Returns:
            The expert's response.
        """
        if self.orchestrator.verbose:
            print(f"\n[Orchestrator] Consulting {expert_name}...")
        
        # Initialize fictional_name for error handling
        fictional_name = expert_name
        
        try:
            # Check if it's a standard personality, otherwise create a dynamic one
            if expert_name.lower() in self.orchestrator.personality_manager.personalities:
                if expert_name.lower() == 'synthesizer':
                        personality = self.orchestrator.personality_manager.get_personality('synthesizer')
                        agent = ModularAgent(provider=self.orchestrator.provider, personality='synthesizer', verbose=False)
                        one_liner = personality.one_liner or f"Expert: {expert_name}"
                        fictional_name = personality.fictional_name or expert_name
                else:
                    personality = self.orchestrator.personality_manager.get_personality(expert_name.lower())
                    agent = ModularAgent(provider=self.orchestrator.provider, personality=expert_name.lower(), verbose=False)
                    one_liner = personality.one_liner or f"Famous figure: {expert_name}"
                    fictional_name = personality.fictional_name or expert_name
            else:
                # Dynamic personality
                personality, usage = await self.orchestrator.personality_manager.create_dynamic_personality(expert_name, self.orchestrator._one_liner_provider, theme=self.orchestrator.theme)
                
                # Log usage for dynamic personality generation
                if usage['total_tokens'] > 0:
                    self.orchestrator._log_usage_background(usage, model=self.orchestrator.model_name, prompt=f"Dynamic personality generation for {expert_name}", response=f"One-liner: {personality.one_liner}")

                agent = ModularAgent(provider=self.orchestrator.provider, personality='default', verbose=False)
                agent.current_personality = personality
                one_liner = personality.one_liner or f"Famous figure: {expert_name}"
                fictional_name = personality.fictional_name or expert_name
            
            # Store mapping for global sanitization
            self.orchestrator.active_experts[expert_name] = fictional_name
            
            # Sanitize question for the event (so frontend doesn't see real names in the question text)
            sanitized_question = sanitize_response(question, self.orchestrator.active_experts, expert_name, fictional_name)
            
            # Store sanitized question for later use in slide creation
            self.orchestrator.pending_questions[fictional_name] = sanitized_question
            
            # Send consult_start event with real name and fictional name
            await self.orchestrator._emit_event({
                "type": "consult_start",
                "expert": expert_name,  # Real name for backend
                "fictional_name": fictional_name,  # Display name for frontend
                "question": sanitized_question,
                "one_liner": one_liner
            })

            response, usage = await agent.run(question)
            self.orchestrator._log_usage_background(usage, model=self.orchestrator.model_name, prompt=question, response=response)
            
            if self.orchestrator.verbose:
                print(f"[{expert_name}] {response[:100]}..." if len(response) > 100 else f"[{expert_name}] {response}")
            
            # Sanitize response before sending to frontend
            sanitized_response = sanitize_response(response, self.orchestrator.active_experts, expert_name, fictional_name)
            
            await self.orchestrator._emit_event({
                "type": "consult_end",
                "expert": expert_name,  # Real name for backend
                "fictional_name": fictional_name,  # Display name for frontend
                "response": sanitized_response,
                "one_liner": one_liner  # Include for consistency
            })
            
            # Create agent_response slide
            if self.orchestrator.db_manager and self.orchestrator.session_id:
                # Generate temporary ID first for immediate event emission
                import uuid
                temp_id = str(uuid.uuid4())
                
                slide = {
                    "type": "agent_response",
                    "sender": fictional_name,
                    "content": sanitized_response,
                    "question": self.orchestrator.pending_questions.get(fictional_name, ""),
                    "oneLiner": one_liner,
                    "id": temp_id  # Temporary ID, will be updated by background task
                }
                
                # Create slide in DB in background (non-blocking)
                async def _create_slide_background():
                    try:
                        slide_id = await self.orchestrator.db_manager.add_slide(self.orchestrator.session_id, slide)
                        # Update slide ID if different from temp (though usually same)
                        if slide_id != temp_id:
                            slide["id"] = slide_id
                    except Exception as e:
                        if self.orchestrator.verbose:
                            print(f"[Orchestrator] Error creating slide in DB: {e}")
                        # Keep temporary ID if DB write fails
                
                # Start background task for slide creation (fire-and-forget)
                asyncio.create_task(_create_slide_background())
                
                # Emit slide_added event immediately (non-blocking)
                await self.orchestrator._emit_event({
                    "type": "slide_added",
                    "slide": slide
                })
                
            # Return the REAL name to the orchestrator so it knows who it talked to
            
            # Emit thinking event again as we return to orchestrator
            await self.orchestrator._emit_event({
                "type": "orchestrator_thinking",
                "label": "Thinking Cats are planning..."
            })

            return f"[{expert_name}]: {response}"
        except Exception as e:
            error_msg = f"Error consulting {expert_name}: {str(e)}"
            await self.orchestrator._emit_event({
                "type": "error",
                "expert": expert_name,
                "message": error_msg
            })
            return error_msg

    async def ask_clarification(self, question: str, options: List[str] = None) -> str:
        """
        Asks the user a clarifying question when the initial problem statement is insufficient.
        
        Args:
            question: The clarifying question to ask the user.
            options: A list of 2-4 creative options or assumptions for the user to choose from.
            
        Returns:
            The user's answer to the question.
        """
        if self.orchestrator.verbose:
            print(f"\n[Orchestrator] Asking for clarification: {question} (Options: {options})")

        
        # Ensure options is a standard list (fix for protobuf RepeatedComposite)
        safe_options = list(options) if options else []
        
        # Update session status to waiting_for_input with question and options
        # No slide is created - clarification is handled via session state
        if self.orchestrator.db_manager and self.orchestrator.session_id:
            await self.orchestrator.db_manager.update_session_status(
                self.orchestrator.session_id, 
                "waiting_for_input", 
                {"type": "clarification", "question": question, "options": safe_options}
            )

        
        await self.orchestrator._emit_event({
            "type": "clarification_request",
            "question": question,
            "options": safe_options
        })
        
        # In a real interactive loop, we would wait for input here.
        # We interrupt the agent loop to wait for user input
        raise AgentInterruption("WAITING_FOR_USER_INPUT")
