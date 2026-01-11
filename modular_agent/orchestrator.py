from typing import List, Dict, Optional, Callable, Any
import google.generativeai as genai
import asyncio
import json
import re
import datetime
from .core import ModularAgent
from .personalities import PersonalityManager
from .llm import GeminiProvider
from .database import DatabaseManager
from .tools.definitions import ToolDefinitions

class Orchestrator:
    def __init__(self, model_name: str, verbose: bool = True, event_callback: Optional[Callable[[Dict], None]] = None, theme: str = "cat", session_id: Optional[str] = None, user_id: Optional[str] = None, db_manager: Optional[DatabaseManager] = None):
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
        self._one_liner_provider = GeminiProvider(model_name="gemini-2.5-flash")
        
        # Initialize tool definitions
        self.tool_definitions = ToolDefinitions(self)
        
        # Plan storage for current execution
        # Simplified schema: {"step": int, "phase": str, "expert": str, "question": str, "format": str}
        self.current_plan = []
        self.plan_executed = []  # List of expert names already executed

        # Initialize the main orchestrator agent (planner)
        # Only has ask_clarification tool - planning phase decides experts
        self.agent = ModularAgent(
            provider=self.provider,
            personality="moderator",
            tools=[],
            verbose=verbose
        )





    def _run_db_task_background(self, coro, task_name: str = "DB operation"):
        """Run a database operation in background with error handling."""
        async def _wrapper():
            try:
                await coro
            except Exception as e:
                if self.verbose:
                    print(f"[Orchestrator] Background {task_name} failed: {e}")
        
        asyncio.create_task(_wrapper())

    def _log_usage_background(self, usage: Dict[str, int], model: str, prompt: str = None, response: str = None):
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
                
                # Update session token usage
                if self.session_id:
                    await self.db_manager.update_session_usage(
                        self.session_id,
                        usage['input_tokens'],
                        output_tokens,
                        thinking_tokens,
                        cost
                    )
                
                # Log detailed LLM call (synchronous, non-blocking)
                self.db_manager.log_llm_call({
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
                import uuid
                temp_id = str(uuid.uuid4())
                
                slide = {
                    "type": "user_message",
                    "question": question,
                    "answer": content,
                    "id": temp_id
                }
                
                # Emit slide_added event immediately with temp ID
                await self._emit_event({
                    "type": "slide_added",
                    "slide": slide
                })
                
                # Create slide in DB in background (non-blocking)
                async def _create_slide():
                    try:
                        slide_id = await self.db_manager.add_slide(self.session_id, slide)
                        if slide_id != temp_id:
                            slide["id"] = slide_id
                        if self.verbose:
                            print(f"[Orchestrator] Created Q&A user_message slide for clarification response")
                    except Exception as e:
                        if self.verbose:
                            print(f"[Orchestrator] Error creating Q&A slide for clarification response: {e}")
                
                self._run_db_task_background(_create_slide(), "Q&A slide creation")
                
                # Emit event to notify frontend
                await self._emit_event({
                    "type": "clarification_answered",
                    "selectedOption": content
                })
            
            # Continue to let the agent process the answer
        else:
            import uuid
            temp_id = str(uuid.uuid4())
            
            slide = {
                "type": "user_message",
                "content": content,
                "id": temp_id
            }
            
            # Emit slide_added event immediately with temp ID
            await self._emit_event({
                "type": "slide_added",
                "slide": slide
            })
            
            # Create slide in DB in background (non-blocking)
            async def _create_slide():
                try:
                    slide_id = await self.db_manager.add_slide(self.session_id, slide)
                    if slide_id != temp_id:
                        slide["id"] = slide_id
                except Exception as e:
                    if self.verbose:
                        print(f"[Orchestrator] Error creating slide: {e}")
            
            self._run_db_task_background(_create_slide(), "user message slide creation")

    async def _emit_event(self, event_data: Dict):
        """Helper method to emit events via callback only (ephemeral SSE, no DB storage)."""
        # Add timestamp if not present
        if "timestamp" not in event_data:
            import time
            event_data["timestamp"] = time.time()

        # Emit event immediately to callback
        if self.event_callback:
            # Check if callback is async
            if asyncio.iscoroutinefunction(self.event_callback):
                await self.event_callback(event_data)
            else:
                self.event_callback(event_data)

    async def generate_plan(self, user_input: str) -> List[Dict[str, str]]:
        """
        Generates an execution plan with a list of experts and their questions.
        Uses a two-phase approach: brainstorming (structuring) vs execution (implementation).
        
        Uses current query and previous questions from all previous plans to avoid repetition.
        
        Returns:
            List of dicts with format: {"expert": str, "question": str, "reason": str}
        """
        if self.verbose:
            print(f"\n[Orchestrator] Generating plan for input: {user_input}")
        
        # Emit thinking event
        await self._emit_event({
            "type": "orchestrator_thinking",
            "label": "Thinking Cats are planning..."
        })
        
        # Fetch conversation history for context (exclude last message if it's the current input)
        conversation_history = await self._get_conversation_context(limit=20)
        
        # Build context string - ensure we don't repeat the current input which is already the first line
        if conversation_history:
            # Simple check: if the last line of history is the current input, strip it
            lines = conversation_history.strip().split("\n")
            if lines and (lines[-1].endswith(user_input[:50]) or user_input[:50] in lines[-1]):
                conversation_history = "\n".join(lines[:-1])
            
            context_section = f"\n\nConversation History (Context for follow-up):\n{conversation_history}\n"
        else:
            context_section = ""
        
        # Fetch previous questions from slides (not questions_answered array)
        previous_questions = []
        if self.db_manager and self.session_id:
            previous_questions = await self.db_manager.get_answered_questions(
                self.session_id, limit=50
            )
        
        # Build context about previous questions to avoid repetition
        questions_context = ""
        if previous_questions:
            questions_text = "\n".join(f"- {q}" for q in previous_questions[-50:])  # Last 50 questions
            questions_context = f"\n\nQuestions already covered in previous plans (do not repeat unless new angle):\n{questions_text}\n"
        
        # Get current date for the prompt
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        
        # Construct prompt for plan generation
        base_prompt = (
            f"Current User Input: {user_input}{context_section}{questions_context}\n\n"
            f"Current Date: {current_date}\n\n"
            "You are a master planner orchestrating a team of famous experts to tackle complex problems. "
            "Your job is to be a RADICAL THINKING PARTNER.\n"
            "To fight average ideas, you must aggressively explore the edges of the problem space.\n\n"
            
            "PHASE: INPUTS (Layer 1 - Ground Truth)\n"
            "Gather foundational information: facts, risks, opportunities, constraints, market signals.\n"
            "CRITICAL: Agents in this phase have WEB SEARCH capabilities.\n"
            "Use when you need real-time information, market data, or external facts.\n"
            "  * Example: 'Research current AI startup funding trends'\n\n"
            
            "PHASE: JUDGMENT (Layer 2 - Wisdom & Patterns)\n"
            "Apply deep human wisdom: instincts, experience, and historical patterns.\n"
            "  * Example: 'What intuitive red flags suggest this business model won't work?'\n\n"
            
            "PHASE: FILTERS (Layer 3 - Ethics & Time)\n"
            "Apply decision filters: principles/values, time horizon, reversibility, second-order effects.\n"
            "  * Example: 'How does this align with sustainable business principles over 10 years?'\n\n"
            
            "PHASE: REALITY_CHECK (Layer 4 - Feasibility)\n"
            "Validate against reality: incentives, power dynamics, execution capacity.\n"
            "  * Example: 'Who has incentives to block this, and how do we navigate them?'\n\n"
     
            "PHASE: DIVERGENCE (Final Layer - Creativity)\n"
            "Use to break mental models and explore 'what if' scenarios.\n"
            "  * Example: 'How would a civilization with infinite energy solve this?'\n"
            "  * Example: 'Propose a solution that seems illegal but isn't.'\n\n"
        

            "Return a JSON array. Each step MUST have:\n"
            '- "step": Step number (1, 2, 3...)\n'
            '- "phase": One of: "inputs", "divergence", "judgment", "filters", "reality_check", "synthesis"\n'
            '- "expert": Famous personality or expert best suited (real name like "Steve Jobs")\n'
            '- "fictional_name": Cat-themed playful name (e.g., "Steve Meows")\n'
            '- "role": Two-word expertise description\n'
            '- "question": Specific question or research objective\n'
            '- "format": Output format that fits the question (e.g., "mermaid flowchart", "comparison table", "sequence diagram", "decision matrix", "pros/cons table", "numbered steps", "code block")\n\n'
            
            "IMPORTANT RULES:\n"
            "- Choose domain experts with unique perspectives on the topic.\n"
            "- Don't use all phases if not needed - be efficient.\n\n"
            
            "Return ONLY the JSON array."
        )
        
        # Clear agent context to avoid duplication since we provide full history in prompt
        self.agent.context.clear()
        
        max_retries = 2
        current_retry = 0
        current_prompt = base_prompt
        plan = []
        
        while current_retry <= max_retries:
            if current_retry > 0:
                if self.verbose:
                    print(f"[Orchestrator] Retrying plan generation (attempt {current_retry}/{max_retries})")
            
            response, usage = await self.agent.chat(current_prompt)
            self._log_usage_background(usage, model=self.model_name, prompt=current_prompt, response=response)
            
            # Check if agent requested clarification (interruption)
            if response == "WAITING_FOR_USER_INPUT":
                if self.verbose:
                    print("[Orchestrator] Agent requested clarification - returning empty plan")
                self.current_plan = []
                self.plan_executed = []
                # Fallback: Treat as no plan, but since we removed tool, this shouldn't happen unless model hallucinates
                return []
            
            # Parse the JSON plan from response
            plan = self._parse_json_plan(response)
            
            if plan:
                if self.verbose:
                    print(f"[Orchestrator] Successfully generated plan with {len(plan)} experts")
                break
            else:
                current_retry += 1
                if current_retry <= max_retries:
                    # Provide feedback for retry
                    current_prompt = (
                        f"{base_prompt}\n\n"
                        f"PREVIOUS ATTEMPT FAILED TO PROVIDE VALID JSON. "
                        f"Response was: {response[:500]}\n"
                        f"Please ensure you return ONLY a valid JSON array of expert objects."
                    )
        
        if not plan and current_retry > max_retries:
            if self.verbose:
                print("[Orchestrator] Failed to generate a valid plan after retries")
            await self._emit_event({
                "type": "orchestrator_error",
                "phase": "planning",
                "error": "Failed to generate a valid plan after multiple attempts."
            })
        
        # Store plan and reset execution tracking
        self.current_plan = plan
        self.plan_executed = []
        
        # Emit plan generated event
        await self._emit_event({
            "type": "plan_generated",
            "plan": plan
        })
        
        return plan

    # Valid phases for multi-layer decision framework
    VALID_PHASES = {"inputs", "divergence", "judgment", "filters", "reality_check", "synthesis"}

    def _parse_json_plan(self, response: str) -> List[Dict[str, str]]:
        """Parses and validates the JSON plan from the model response.
        
        Expected schema per item:
        - step: int (step number)
        - phase: str ("brainstorming" or "execution")
        - expert: str (real expert name)
        - fictional_name: str (cat-themed fictional name)
        - role: str (two-word role description)
        - question: str (the question for the expert)
        - format: str (expected output format)
        """
        plan = []
        try:
            # Try to extract JSON from response (might have markdown blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            
            # Clean up any potential leading/trailing non-JSON text
            json_match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            
            parsed_items = json.loads(json_str)
            
            if not isinstance(parsed_items, list):
                if self.verbose:
                    print(f"[Orchestrator] Plan is not a list: {type(parsed_items)}")
                return []

            # Validate plan structure and add valid items
            for idx, item in enumerate(parsed_items):
                if not isinstance(item, dict):
                    if self.verbose:
                        print(f"[Orchestrator] Skipping non-dict plan item: {item}")
                    continue
                
                # Required fields: expert, fictional_name, role, and question
                if "expert" not in item or "question" not in item or "fictional_name" not in item or "role" not in item:
                    if self.verbose:
                        print(f"[Orchestrator] Skipping item missing required fields: {item}")
                    continue
                
                # Ensure step number exists (auto-assign if missing)
                if "step" not in item:
                    item["step"] = idx + 1
                
                # Validate and default phase
                if "phase" not in item or item["phase"] not in self.VALID_PHASES:
                    if self.verbose and "phase" in item:
                        print(f"[Orchestrator] Invalid phase '{item.get('phase')}', defaulting to 'execution'")
                    item["phase"] = "execution"
                
                # Ensure format exists
                if "format" not in item:
                    item["format"] = "structured response"
                
                plan.append(item)
            
            return plan
        except (json.JSONDecodeError, Exception) as e:
            if self.verbose:
                print(f"[Orchestrator] Failed to parse plan JSON: {e}")
            return []

    async def execute_plan(self, user_input: str = "") -> str:
        """
        Executes the current plan by consulting each expert in sequence.
        Before each expert call, generates a self-contained question that includes
        all necessary context from the original request and previous expert contributions.
        
        Args:
            user_input: The original user request for context
        
        Returns:
            Summary of execution status
        """
        if not self.current_plan:
            if self.verbose:
                print("[Orchestrator] No plan to execute")
            return "No plan generated"
        
        if self.verbose:
            print(f"\\n[Orchestrator] Executing plan with {len(self.current_plan)} steps")
        
        # Emit execution started event
        await self._emit_event({
            "type": "plan_execution_started",
            "total_steps": len(self.current_plan)
        })
        
        # Track accumulated context from previous experts
        accumulated_context = []  # List of {"expert": str, "objective": str, "response": str}
        
        # Execute each step in the plan
        for idx, step_item in enumerate(self.current_plan):
            expert_name = step_item.get("expert", "")
            base_question = step_item.get("question", "")
            fictional_name = step_item.get("fictional_name", expert_name)  # Fallback to real name
            role = step_item.get("role", "Expert")  # Fallback to generic role
            
            # Two-phase fields
            step_num = step_item.get("step", idx + 1)
            phase = step_item.get("phase", "execution")
            output_format = step_item.get("format", "structured response")
            
            if not expert_name or not base_question:
                if self.verbose:
                    print(f"[Orchestrator] Skipping invalid plan step: {step_item}")
                continue
            
            try:
                # Build structured prompt with context (no LLM call, just template)
                enhanced_question = self._build_expert_prompt(
                    question=base_question,
                    output_format=output_format,
                    user_input=user_input,
                    previous_context=accumulated_context
                )
                
                # Emit progress event with phase context
                await self._emit_event({
                    "type": "plan_step_executing",
                    "step": step_num,
                    "total": len(self.current_plan),
                    "expert": expert_name,
                    "phase": phase
                })
                
                # Use research agent for inputs phase, regular expert for others
                if phase == "inputs":
                    response = await self.tool_definitions.consult_research_agent(
                        expert_name, 
                        enhanced_question,
                        display_question=base_question,
                        fictional_name=fictional_name,
                        role=role
                    )
                else:
                    # Consult the expert with the enhanced prompt, but pass original question for display
                    response = await self.tool_definitions.consult_expert(
                        expert_name, 
                        enhanced_question,
                        display_question=base_question,  # Original simple question for UI
                        fictional_name=fictional_name,   # From plan
                        role=role                        # From plan
                    )
                
                # Extract just the response content (remove "[expert_name]: " prefix if present)
                response_content = response
                if response.startswith(f"[{expert_name}]:"):
                    response_content = response[len(f"[{expert_name}]:"):].strip()
                
                # Add this expert's contribution to accumulated context
                accumulated_context.append({
                    "expert": expert_name,
                    "phase": phase,
                    "response": response_content[:500]  # Truncate to keep context manageable
                })
                
                # Track execution
                self.plan_executed.append(expert_name)
                
                # Emit progress event
                await self._emit_event({
                    "type": "plan_step_completed",
                    "step": step_num,
                    "total": len(self.current_plan),
                    "expert": expert_name,
                    "phase": phase
                })
                
            except Exception as e:
                if self.verbose:
                    print(f"[Orchestrator] Error executing plan step {step_num} for {expert_name}: {e}")
                
                await self._emit_event({
                    "type": "plan_step_error",
                    "step": step_num,
                    "expert": expert_name,
                    "phase": phase,
                    "error": str(e)
                })
        
        # Emit execution completed event
        await self._emit_event({
            "type": "plan_execution_completed",
            "executed": len(self.plan_executed),
            "total": len(self.current_plan)
        })
        
        return f"Executed plan with {len(self.plan_executed)}/{len(self.current_plan)} steps"

    def _build_expert_prompt(
        self,
        question: str,
        output_format: str,
        user_input: str,
        previous_context: List[Dict[str, str]]
    ) -> str:
        """
        Builds a structured prompt for an expert using a simple template.
        No LLM call - just efficient string formatting.
        
        Args:
            question: The question for this expert
            output_format: Expected output format (e.g., "bullet list")
            user_input: Original user request
            previous_context: List of previous expert contributions
            
        Returns:
            A structured prompt string
        """
        parts = []
        
        # Section 1: Original request (brief context)
        if user_input:
            truncated_input = user_input[:300] + "..." if len(user_input) > 300 else user_input
            parts.append(f"Request: {truncated_input}")
        
        # Section 2: Prior insights (if any) - just key points
        if previous_context:
            insights = []
            for ctx in previous_context[-4:]:  # Only last 3 experts max
                response = ctx.get("response", "")
                first_sentence = response.split('.')[0][:150] if response else ""
                if first_sentence:
                    insights.append(f"• {ctx['expert']}: {first_sentence}")
            if insights:
                parts.append("Prior insights:\n" + "\n".join(insights))
        
        # Section 3: The actual question
        parts.append(f"Question: {question}")
        
        # Section 4: Expected format
        if output_format:
            parts.append(f"Format: {output_format}")
        
        return "\n\n".join(parts)

    async def process(self, user_input: str) -> str:
        """
        Two-phase process:
        1. PLANNING PHASE: Generate a plan with list of experts
        2. EXECUTION PHASE: Execute the plan by consulting each expert
        
        Returns a summary of execution.
        """
        if self.verbose:
            print(f"\\n[Orchestrator] Received input: {user_input}")
            
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
        
        # Update status to processing (non-blocking)
        if self.db_manager and self.session_id:
            self._run_db_task_background(
                self.db_manager.update_session_status(self.session_id, "processing", None),
                "session status update (processing)"
            )

        # Create user_message slide for the input
        await self._create_user_message_slide(user_input, is_clarification_response=is_clarification_response, pending_interaction=pending_interaction)

        # ===== PHASE 1: PLANNING =====
        # Generate the execution plan (experts + questions + reasons)
        try:
            plan = await self.generate_plan(user_input)
        except Exception as e:
            if self.verbose:
                print(f"[Orchestrator] Error generating plan: {e}")
            
            await self._emit_event({
                "type": "orchestrator_error",
                "phase": "planning",
                "error": str(e)
            })
            
            if self.db_manager and self.session_id:
                self._run_db_task_background(
                    self.db_manager.update_session_status(self.session_id, "idle"),
                    "session status update (idle after error)"
                )
            
            return f"Error generating plan: {str(e)}"
        
        # If plan is empty or no experts were chosen, check if clarification was requested
        if not plan:
            if self.verbose:
                print("[Orchestrator] No experts in plan")
            
            # Check if session is waiting for input (clarification was requested)
            if self.db_manager and self.session_id:
                session = await self.db_manager.get_session(self.session_id)
                if session and session.get("status") == "waiting_for_input":
                    # Clarification was requested - keep status as waiting_for_input
                    if self.verbose:
                        print("[Orchestrator] Clarification requested - keeping status as waiting_for_input")
                    return "Waiting for user clarification"
                else:
                    # No clarification and no plan - this might be a failure or just no experts needed
                    # If it was a failure, generate_plan already emitted an error event
                    self._run_db_task_background(
                        self.db_manager.update_session_status(self.session_id, "idle"),
                        "session status update (idle - no experts)"
                    )
            
            return "No experts were selected for this query. Please try being more specific."
        
        # ===== PHASE 2: EXECUTION =====
        # Execute the plan by consulting each expert with full context
        try:
            result = await self.execute_plan(user_input=user_input)
        except Exception as e:
            if self.verbose:
                print(f"[Orchestrator] Error executing plan: {e}")
            
            await self._emit_event({
                "type": "orchestrator_error",
                "phase": "execution",
                "error": str(e)
            })
            
            result = f"Error during execution: {str(e)}"
        
        # Set session to idle after execution completes (non-blocking)
        if self.db_manager and self.session_id:
            self._run_db_task_background(
                self.db_manager.update_session_status(self.session_id, "idle"),
                "session status update (idle after execution)"
            )
        
        return result
 
    async def _get_conversation_context(self, limit: int = 15) -> str:
        """
        Retrieves recent conversation context from session slides.
        Returns a formatted string of user inputs and expert questions.
        """
        if not self.db_manager or not self.session_id:
            return ""
            
        try:
            session = await self.db_manager.get_session(self.session_id)
            if not session or "slides" not in session:
                return ""
                
            slides = session.get("slides", [])
            context_items = []
            
            # Process slides in order
            for slide in slides:
                slide_type = slide.get("type")
                
                if slide_type == "user_message":
                    if "question" in slide and "answer" in slide:
                        # Q&A format
                        context_items.append(f"User (Clarification): {slide['question']} -> {slide['answer']}")
                    elif "content" in slide:
                        context_items.append(f"User: {slide['content']}")
                        
                elif slide_type == "agent_response":
                    expert = slide.get("sender", "Expert")
                    question = slide.get("question", "")
                    if question:
                        context_items.append(f"Expert {expert} was asked: {question}")
            
            # Return last N items
            if not context_items:
                return ""
                
            return "\\n".join(context_items[-limit:])
            
            
        except Exception as e:
            if self.verbose:
                print(f"[Orchestrator] Error getting conversation context: {e}")
            return ""
 