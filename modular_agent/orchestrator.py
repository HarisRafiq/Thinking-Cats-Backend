from typing import List, Dict, Optional, Callable, Any
import google.generativeai as genai
import asyncio
import json
import re
from .core import ModularAgent
from .personalities import PersonalityManager
from .llm import GeminiProvider
from .database import DatabaseManager
from .tools.definitions import ToolDefinitions
from .utils.sanitization import sanitize_response

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
        # Schema: {"step": int, "expert": str, "fictional_name": str, "role": str, "question": str, "format": str}
        self.current_plan = []
        self.plan_executed = []  # List of expert names already executed
        self.suggested_questions = []  # List of suggested follow-up questions from current plan

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
        
        The orchestrator decomposes complex goals into key components and assigns domain experts
        to explore each component deeply using appropriate thinking frameworks. The plan is designed
        such that all expert responses collectively complete the task - convergence happens through
        intelligent planning, not post-hoc synthesis.
        
        Uses current query and previous questions from all previous plans to avoid repetition.
        
        Returns:
            List of dicts with format: {"expert": str, "fictional_name": str, "role": str, "question": str, "format": str}
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
        
        # Construct prompt for plan generation
        base_prompt = (
            f"Current User Input: {user_input}{context_section}{questions_context}\n\n"
            "You are a meta-thinking orchestrator. Available thinking frameworks:\n"
            "• First Principles\n"
            "• Critical Thinking\n"
            "• Socratic Method\n"
            "• Parallel Thinking (Six Thinking Hats)\n"
            "• Design Thinking\n"
            "• Red Team/Blue Team\n"
            "• Systems Thinking\n"
            "• Lateral Thinking\n\n"
            
            "Your task:\n"
            "1. Decompose complex requests into KEY COMPONENTS that need deep exploration\n"
            "2. For each component, select the best domain expert\n"
            "3. Ask CLEAN, FOCUSED questions - let experts apply their natural approach\n"
            "4. Internally note which framework fits (for system use), but keep questions simple\n\n"
            
            "PRINCIPLES:\n"
            "- Decompose goals into comprehensive components\n"
            "- Select experts by domain expertise\n"
            "- Ask clear questions without over-prescribing the approach\n"
            "- Plan collectively completes the task\n\n"
            
            "Simple queries: 1 expert, 1 focused question\n"
            "Complex queries: 3-5 experts, each with a focused question on their component\n"
            "Format: Specify visual formats like 'bulleted list', 'comparison table', 'step-by-step guide', or 'short manifesto'.\n\n"
            
            "Examples:\n\n"
            
            "Query: 'Explain quantum entanglement' [SIMPLE]\n"
            '[\n'
            '  {"expert": "Richard Feynman", "fictional_name": "Richard Furrman", "role": "Physics Explainer", '
            '   "question": "What is quantum entanglement and why does it matter?", '
            '   "format": "explanation with diagrams"}\n'
            "]\n\n"
            
            "Query: 'Create a business plan for an AI tutoring app' [COMPLEX]\n"
            '[\n'
            '  {"expert": "Clayton Christensen", "fictional_name": "Clayton Whiskersten", "role": "Innovation Theorist", '
            '   "question": "What job are students and parents really hiring an AI tutor to do?", '
            '   "format": "market analysis with comparison table"},\n'
            '  {"expert": "Andrew Ng", "fictional_name": "Andrew Meow", "role": "AI Educator", '
            '   "question": "How should the AI tutoring system be designed to actually improve learning outcomes?", '
            '   "format": "system architecture with mermaid diagram"},\n'
            '  {"expert": "Ben Thompson", "fictional_name": "Ben Thompurr", "role": "Strategy Analyst", '
            '   "question": "What business model would make this AI tutor defensible and profitable?", '
            '   "format": "business model canvas with projections"},\n'
            '  {"expert": "April Dunford", "fictional_name": "April Pawford", "role": "Positioning Expert", '
            '   "question": "How should we position and launch this to acquire the first 1000 customers?", '
            '   "format": "GTM plan with timeline"}\n'
            "]\n\n"
            
            "Query: 'Should I quit my job to start a company?' [COMPLEX]\n"
            '[\n'
            '  {"expert": "Paul Graham", "fictional_name": "Pawl Graham", "role": "Startup Mentor", '
            '   "question": "What indicates someone is ready to start a company vs should wait?", '
            '   "format": "readiness assessment"},\n'
            '  {"expert": "Nassim Taleb", "fictional_name": "Nassim Taileb", "role": "Risk Philosopher", '
            '   "question": "How should someone structure this career transition to be antifragile?", '
            '   "format": "risk analysis with scenarios"},\n'
            '  {"expert": "Naval Ravikant", "fictional_name": "Naval Meowvikant", "role": "Wealth Philosopher", '
            '   "question": "What\'s the financial reality of bootstrapping a startup?", '
            '   "format": "financial breakdown with runway calculator"}\n'
            "]\n\n"
            
            "ADDITIONAL TASK - Suggest Follow-up Questions:\n"
            "Additionally, suggest 3 creative follow-up questions the user could ask to continue exploring:\n"
            "- Questions that explore angles NOT covered in this plan\n"
            "- Questions that go deeper into promising areas\n"
            "- Questions that challenge assumptions or consider alternatives\n"
            "- Questions that are specific, actionable, and concise (max 15 words)\n"
            "- Avoid repeating topics already in the current plan or previous conversation\n\n"
            
            "Return a JSON object with TWO fields:\n"
            "{\n"
            '  "plan": [array of expert objects as shown above],\n'
            '  "suggested_questions": ["Question 1?", "Question 2?", "Question 3?"]\n'
            "}\n\n"
            "Required fields in plan items: expert, fictional_name, role, question, format"
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
                        f"Please ensure you return ONLY a valid JSON object with 'plan' and 'suggested_questions' fields."
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
            "plan": plan,
            "suggested_questions": self.suggested_questions
        })
        
        return plan

    def _parse_json_plan(self, response: str) -> List[Dict[str, str]]:
        """Parses and validates the JSON plan from the model response.
        
        Expected schema per item:
        - step: int (step number - auto-assigned if missing)
        - expert: str (real expert name)
        - fictional_name: str (cat-themed fictional name)
        - role: str (two-word role description)
        - question: str (the question for the expert)
        - format: str (expected output format)
        
        New format also extracts suggested_questions if present:
        {
          "plan": [...],
          "suggested_questions": ["Q1?", "Q2?", ...]
        }
        """
        plan = []
        try:
            # Try to extract JSON from response (might have markdown blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            
            # Try to match both object and array patterns
            # First try to match an object
            obj_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            arr_match = re.search(r'\[.*\]', json_str, re.DOTALL)
            
            # Prefer object match if it exists (new format)
            if obj_match:
                json_str = obj_match.group(0)
            elif arr_match:
                json_str = arr_match.group(0)
            
            parsed_data = json.loads(json_str)
            
            # Handle new object format with plan and suggested_questions
            if isinstance(parsed_data, dict):
                if "plan" in parsed_data:
                    parsed_items = parsed_data["plan"]
                    # Extract suggested questions if present
                    if "suggested_questions" in parsed_data and isinstance(parsed_data["suggested_questions"], list):
                        self.suggested_questions = parsed_data["suggested_questions"]
                        if self.verbose:
                            print(f"[Orchestrator] Extracted {len(self.suggested_questions)} suggested questions")
                    else:
                        self.suggested_questions = []
                else:
                    if self.verbose:
                        print(f"[Orchestrator] Object format but no 'plan' field found")
                    return []
            # Handle legacy array format
            elif isinstance(parsed_data, list):
                parsed_items = parsed_data
                self.suggested_questions = []  # No questions in legacy format
            else:
                if self.verbose:
                    print(f"[Orchestrator] Invalid format: {type(parsed_data)}")
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
            step_num = step_item.get("step", idx + 1)
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
                
                # Emit progress event
                await self._emit_event({
                    "type": "plan_step_executing",
                    "step": step_num,
                    "total": len(self.current_plan),
                    "expert": expert_name
                })
                
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
                    "response": response_content[:500]  # Truncate to keep context manageable
                })
                
                # Track execution
                self.plan_executed.append(expert_name)
                
                # Emit progress event
                await self._emit_event({
                    "type": "plan_step_completed",
                    "step": step_num,
                    "total": len(self.current_plan),
                    "expert": expert_name
                })
                
            except Exception as e:
                if self.verbose:
                    print(f"[Orchestrator] Error executing plan step {step_num} for {expert_name}: {e}")
                
                await self._emit_event({
                    "type": "plan_step_error",
                    "step": step_num,
                    "expert": expert_name,
                    "error": str(e)
                })
        
        # Emit execution completed event
        await self._emit_event({
            "type": "plan_execution_completed",
            "executed": len(self.plan_executed),
            "total": len(self.current_plan)
        })
        
        return f"Executed plan with {len(self.plan_executed)}/{len(self.current_plan)} steps"

    async def _update_suggested_questions(self):
        """
        Updates the session's suggested_questions property with accumulated questions,
        filtering out questions that have already been asked by the user.
        """
        if not self.db_manager or not self.session_id:
            return
        
        if not self.suggested_questions:
            return
        
        try:
            session = await self.db_manager.get_session(self.session_id)
            if not session:
                return
            
            # Get existing accumulated questions
            accumulated = session.get("suggested_questions", [])
            
            # Add new questions (avoid duplicates)
            for question in self.suggested_questions:
                if question not in accumulated:
                    accumulated.append(question)
            
            if self.verbose:
                print(f"[Orchestrator] Accumulated {len(accumulated)} total questions")
            
            # Get all user messages to filter out asked questions
            user_messages = []
            slides = session.get("slides", [])
            
            for slide in slides:
                if slide.get("type") == "user_message":
                    if "content" in slide:
                        user_messages.append(slide["content"].strip().lower())
                    if "answer" in slide:
                        user_messages.append(slide["answer"].strip().lower())
            
            # Filter out questions that have been asked
            filtered_questions = []
            for question in accumulated:
                question_lower = question.strip().lower()
                # Check if this question was asked (simple substring match)
                is_asked = any(
                    question_lower in msg or msg in question_lower
                    for msg in user_messages
                )
                if not is_asked:
                    filtered_questions.append(question)
            
            if self.verbose:
                print(f"[Orchestrator] Filtered to {len(filtered_questions)} questions")
            
            # Update session with filtered questions
            await self.db_manager.update_session_suggested_questions(
                self.session_id, 
                filtered_questions
            )
            
            # Emit event with updated questions
            await self._emit_event({
                "type": "suggested_questions_updated",
                "questions": filtered_questions
            })
            
        except Exception as e:
            if self.verbose:
                print(f"[Orchestrator] Error updating suggested questions: {e}")

    def _build_expert_prompt(
        self,
        question: str,
        output_format: str,
        user_input: str,
        previous_context: List[Dict[str, str]]
    ) -> str:
        """
        Builds a clean prompt for an expert using a simple template.
        No LLM call - just efficient string formatting.
        
        Provides context and format guidance without over-prescribing the approach.
        
        Args:
            question: The question for this expert
            output_format: Expected output format (e.g., "explanation with diagrams")
            user_input: Original user request
            previous_context: List of previous expert contributions
            
        Returns:
            A structured prompt string
        """
        parts = []
        
        # Section 1: Original request (brief context)
        if user_input:
            truncated_input = user_input[:300] + "..." if len(user_input) > 300 else user_input
            parts.append(f"Original User Request: {truncated_input}")
        
        # Section 2: Prior insights (IMPROVED)
        if previous_context:
            insights = []
            for ctx in previous_context[-4:]:  # Only last 4 experts max
                response = ctx.get("response", "")
                # Take first 400 chars instead of just first sentence to keep more context
                summary = response[:400].strip() + "..." if len(response) > 400 else response
                if summary:
                    insights.append(f"• {ctx['expert']} previously said: {summary}")
            if insights:
                parts.append("Context from previous experts:\n" + "\n".join(insights))
        
        # Section 3: The actual question
        parts.append(f"Your Specific Question: {question}")
        
        # Section 4: Expected format (IMPROVED)
        parts.append("\nFORMATTING GUIDELINES (CRITICAL):")
        parts.append("1. You are creating content for a PRESENTATION SLIDE.")
        parts.append("2. Be CONCISE. Use bullet points, bold text, and short paragraphs.")
        parts.append("3. Do not use 'I think' or 'In my opinion' filler. State your expertise directly.")
        
        if output_format:
            parts.append(f"4. Desired Output Style: {output_format}")
        
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
        
        # Update suggested questions after execution completes
        await self._update_suggested_questions()
        
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

    async def generate_social_content_plan(self, session_id: str, instruction: str = None) -> List[Dict[str, str]]:
        """
        Generates a plan for social media content based on the session history.
        Returns a list of 6 items, each with a caption and visual description.
        If instruction is provided, uses it to refine the style/content.
        """
        if self.verbose:
            print(f"[Orchestrator] Generating social content plan for session {session_id} (Instruction: {instruction})")
            
        # Get conversation context
        # We need to temporarily set session_id if it's different (though typically Orchestrator is per request)
        # For now, assume self.session_id is already set or we use the passed one
        original_session_id = self.session_id
        self.session_id = session_id
        
        context = await self._get_conversation_context(limit=50)
        self.session_id = original_session_id # Restore
        
        if not context:
            if self.verbose:
                print("[Orchestrator] No context found for social plan")
            return []

    async def generate_social_post(
        self,
        platform: str,
        caption: str,
        visual_description: str
    ) -> str:
        """
        Generates a platform-specific social media post using the provided caption and
        visual description as creative context.

        Args:
            platform: Target platform (e.g., "twitter", "x", "instagram", "linkedin", "facebook").
            caption: Short punchy caption or headline to inspire the post.
            visual_description: Brief description of the accompanying visual.

        Returns:
            Post text tailored for the requested platform.
        """
        try:
            platform_key = (platform or "").strip().lower()
            if platform_key == "x":
                platform_key = "twitter"

            # Platform-specific guidance
            guidance_map = {
                "twitter": (
                    "Write a single tweet (max 280 chars). Keep it crisp,"
                    " witty, and high-signal. Optional 1-2 short hashtags."
                    " Avoid excessive emojis or links. Output ONLY the tweet."
                ),
                "instagram": (
                    "Write an Instagram caption in 1-2 short lines."
                    " Add 4-8 relevant lowercase hashtags at the end."
                    " Keep it aesthetic, concise, and friendly."
                    " Output ONLY the caption body."
                ),
                "linkedin": (
                    "Write a concise LinkedIn post (2-4 sentences)."
                    " Professional tone, actionable insight."
                    " Optionally include 1-2 tasteful hashtags."
                    " Output ONLY the post text."
                ),
                "facebook": (
                    "Write a friendly Facebook post (1-2 sentences)."
                    " Conversational, approachable."
                    " Avoid overusing hashtags or emojis."
                    " Output ONLY the post text."
                ),
            }

            guidance = guidance_map.get(platform_key, (
                "Write a concise social post (1-2 sentences)."
                " Tailor tone to a general audience."
                " Output ONLY the post text."
            ))

            system_instruction = (
                "You are an expert social media copywriter."
                " Craft platform-appropriate, safe, and non-harmful content."
                " Do not include disclaimers, prefaces, or metadata."
                " Avoid offensive or unsafe content."
            )

            prompt = (
                f"Platform: {platform_key}\n"
                f"Caption: {caption.strip()}\n"
                f"Visual: {visual_description.strip()}\n\n"
                f"Instructions: {guidance}\n"
            )

            # Use lightweight provider optimized for short copy
            response = await self._one_liner_provider.generate_content_async(
                prompt=prompt,
                system_instruction=system_instruction
            )

            text = response.text if getattr(response, "text", None) else ""
            usage = self._one_liner_provider.get_token_usage(response)
            # Log usage in background
            self._log_usage_background(
                usage,
                model=self._one_liner_provider.model_name,
                prompt=prompt,
                response=text
            )

            # Minimal post-processing
            cleaned = (text or "").strip()

            # Enforce Twitter length limit defensively
            if platform_key == "twitter" and len(cleaned) > 280:
                # Trim to last full word under limit
                trimmed = cleaned[:280]
                last_space = trimmed.rfind(" ")
                if last_space > 0:
                    cleaned = trimmed[:last_space].strip()
                else:
                    cleaned = trimmed.strip()

            return cleaned

        except Exception as e:
            if self.verbose:
                print(f"[Orchestrator] Error generating social post: {e}")
            # Surface a simple error string upward; router will handle HTTPException
            raise
            
        instruction_text = ""
        if instruction:
            instruction_text = (
                f"\n\nUSER REFINEMENT INSTRUCTION:\n"
                f"The user wants to refine the previous output with this instruction: \"{instruction}\"\n"
                f"Ensure the captions and visual descriptions adhere strictly to this request."
            )

        prompt = (
            f"Analyze the following conversation and extract 6 key insights, quotes, or transformative ideas.\n"
            f"For EACH insight, create a 'Social Card'.\n\n"
            f"Conversation History:\n{context}\n"
            f"{instruction_text}\n"
            f"Task:\n"
            f"1. Identify 6 distinct, high-impact concepts discussed.\n"
            f"2. For each, write a short, punchy, viral-style CAPTION (max 10 words).\n"
            f"3. For each, write a detailed VISUAL DESCRIPTION for an AI image generator. "
            f"The visual should be abstract, high-quality, 3D render style, or minimalistic digital art. "
            f"Avoid text in the image. Focus on metaphors, lighting, and composition.\n\n"
            f"Return ONLY a JSON array of 6 objects:\n"
            '[\n'
            '  {"caption": "Start small.", "visual_description": "A single glowing ember in a dark void, cinematic lighting"},\n'
            '  {"caption": "Think big.", "visual_description": "A vast nebula expanding into the cosmos, 8k resolution"}\n'
            ']'
        )
        
        try:
            # unique usage tracking for this? keeping it simple for now
            response, _ = await self.agent.chat(prompt)
            
            # Parse JSON
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
                
            json_match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                
            plan = json.loads(json_str)
            
            # Validation
            if isinstance(plan, list) and len(plan) > 0:
                # Ensure we have exactly 6 or trim/pad?
                # The prompt asks for 6. Let's return what we got, but capped at 6 for grid logic safety later
                return plan[:6]
                
            return []
            
        except Exception as e:
            if self.verbose:
                print(f"[Orchestrator] Error generating social plan: {e}")
            return []

