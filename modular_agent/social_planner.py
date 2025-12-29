"""
Social Content Planner - Generates viral social media content from chat sessions.

Extracts key insights, quotes, and transformative ideas from expert conversations
and creates engaging social media cards with captions and visual descriptions.
"""

from typing import List, Dict, Optional
import json
import re
from modular_agent.llm import GeminiProvider


class SocialContentPlanner:
    """
    Generates social media content plans from chat session history.
    
    Unlike the orchestrator, this focuses purely on content extraction and
    viral content creation, leveraging the full conversation including expert
    responses that were previously ignored.
    """
    
    def __init__(self, db_manager, model_name: str = "gemini-3-flash", verbose: bool = False):
        """
        Initialize the social content planner.
        
        Args:
            db_manager: Database manager instance for accessing session data
            model_name: LLM model to use for content generation
            verbose: Enable verbose logging
        """
        self.db_manager = db_manager
        self.model_name = model_name
        self.verbose = verbose
        self.llm = GeminiProvider(model_name=model_name)
        
    async def _extract_conversation_content(self, session_id: str, limit: int = 50) -> str:
        """
        Extract rich conversation content from session slides.
        
        Unlike the orchestrator's _get_conversation_context, this includes:
        - User questions
        - Expert names and taglines (for attribution)
        - FULL expert response content (the "gold content" previously ignored)
        
        Args:
            session_id: Session ID to extract content from
            limit: Maximum number of slides to process (most recent)
            
        Returns:
            Formatted conversation string with full context
        """
        try:
            session = await self.db_manager.get_session(session_id)
            if not session or "slides" not in session:
                if self.verbose:
                    print(f"[SocialPlanner] No session or slides found for {session_id}")
                return ""
                
            slides = session.get("slides", [])
            if not slides:
                return ""
            
            # Process most recent slides
            recent_slides = slides[-limit:] if len(slides) > limit else slides
            
            conversation_parts = []
            
            # Add session problem/context if available
            if session.get("problem"):
                conversation_parts.append(f"INITIAL CONTEXT: {session['problem']}\n")
            
            for slide in recent_slides:
                slide_type = slide.get("type")
                
                if slide_type == "user_message":
                    # User questions or clarifications
                    if "question" in slide and "answer" in slide:
                        conversation_parts.append(
                            f"USER CLARIFICATION:\nQ: {slide['question']}\nA: {slide['answer']}\n"
                        )
                    elif "content" in slide:
                        conversation_parts.append(f"USER: {slide['content']}\n")
                        
                elif slide_type == "agent_response":
                    # Expert responses - THE GOLD CONTENT
                    expert = slide.get("sender", "Expert")
                    one_liner = slide.get("oneLiner", "")
                    question = slide.get("question", "")
                    content = slide.get("content", "")
                    
                    # Build expert response block
                    expert_block = f"EXPERT: {expert}"
                    if one_liner:
                        expert_block += f" ({one_liner})"
                    expert_block += "\n"
                    
                    if question:
                        expert_block += f"Asked about: {question}\n"
                    
                    # THIS IS THE KEY DIFFERENCE - we include the actual response content
                    if content:
                        expert_block += f"Response:\n{content}\n"
                    
                    conversation_parts.append(expert_block)
            
            if not conversation_parts:
                return ""
                
            return "\n---\n".join(conversation_parts)
            
        except Exception as e:
            if self.verbose:
                print(f"[SocialPlanner] Error extracting conversation content: {e}")
            return ""
    
    async def generate_content_plan(
        self, 
        session_id: str, 
        instruction: Optional[str] = None,
        num_cards: int = 6
    ) -> List[Dict[str, str]]:
        """
        Generate viral social media content plan from session conversation.
        
        Creates social cards with punchy captions and visual descriptions,
        focusing on the most shareable, quotable, and transformative moments
        from the expert consultations.
        
        Args:
            session_id: Session to generate content from
            instruction: Optional refinement instruction (e.g., "make it more professional")
            num_cards: Number of social cards to generate (default 6)
            
        Returns:
            List of dicts with 'caption', 'visual_description', and optionally 'expert_source'
        """
        if self.verbose:
            print(f"[SocialPlanner] Generating content plan for session {session_id}")
            
        # Extract full conversation content (including expert responses)
        conversation = await self._extract_conversation_content(session_id, limit=50)
        
        if not conversation:
            if self.verbose:
                print("[SocialPlanner] No conversation content found")
            return []
        
        # Build prompt for viral content extraction
        instruction_text = ""
        if instruction:
            instruction_text = (
                f"\n\nREFINEMENT INSTRUCTION:\n"
                f"{instruction}\n"
                f"Apply this instruction to the style, tone, or focus of the content.\n"
            )
        
        prompt = f"""You are a viral social media content strategist. Your task is to analyze a deep conversation between a user and expert advisors, then extract the most shareable, memorable content.

CONVERSATION HISTORY:
{conversation}

{instruction_text}

TASK:
Create {num_cards} social media cards that capture the most viral-worthy moments from this conversation.

Focus on:
1. **Quotable insights** - Memorable one-liners that make people think
2. **Counter-intuitive wisdom** - Ideas that challenge conventional thinking
3. **Actionable frameworks** - Practical mental models people can apply
4. **Transformation moments** - Before/after realizations
5. **Expert authority** - Leverage the expert's credibility and unique perspective
6. **Emotional resonance** - Content that sparks curiosity, inspiration, or revelation

STYLE GUIDELINES:
- Captions should be punchy, provocative, and shareable (5-15 words max)
- Visual descriptions should be metaphorical, cinematic, and evocative
- Think premium digital art: dramatic lighting, 8k resolution, artstation trending
- Avoid literal representations - use abstract concepts and visual metaphors
- Each card should stand alone as a powerful piece of content

OUTPUT FORMAT:
Return a JSON array of exactly {num_cards} objects:
[
  {{
    "caption": "The best advice you'll ignore until it's too late",
    "visual_description": "A glowing fork in a dark path, one side fading into shadows, cinematic lighting, 8k, moody atmosphere",
    "expert_source": "Steve Jobs (Design Principles)"
  }}
]

Ensure valid JSON. No markdown formatting. Start directly with ["""

        try:
            # Generate content using LLM
            response = await self.llm.generate(
                prompt=prompt,
                system_instruction="You are an expert at extracting viral social media content. Always return valid JSON."
            )
            
            if self.verbose:
                print(f"[SocialPlanner] LLM Response: {response[:200]}...")
            
            # Parse JSON response
            json_str = response.strip()
            
            # Clean up common JSON formatting issues
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            # Extract JSON array
            json_match = re.search(r'\[.*\]', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            
            plan = json.loads(json_str)
            
            # Validate and return
            if isinstance(plan, list) and len(plan) > 0:
                # Ensure we have the requested number of cards (or close to it)
                if len(plan) < num_cards and self.verbose:
                    print(f"[SocialPlanner] Warning: Generated {len(plan)} cards, requested {num_cards}")
                
                return plan[:num_cards]
            
            if self.verbose:
                print("[SocialPlanner] Failed to parse valid plan from LLM response")
            return []
            
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"[SocialPlanner] JSON parsing error: {e}")
                print(f"[SocialPlanner] Raw response: {response}")
            return []
        except Exception as e:
            if self.verbose:
                print(f"[SocialPlanner] Error generating content plan: {e}")
                import traceback
                traceback.print_exc()
            return []
    
    async def refine_content_plan(
        self,
        session_id: str,
        instruction: str,
        num_cards: int = 6
    ) -> List[Dict[str, str]]:
        """
        Refine content plan based on user instruction.
        
        This is essentially an alias for generate_content_plan with an instruction,
        but exists for API clarity.
        
        Args:
            session_id: Session to generate content from
            instruction: How to refine the content (e.g., "make it more technical")
            num_cards: Number of cards to generate
            
        Returns:
            List of refined social card dicts
        """
        if self.verbose:
            print(f"[SocialPlanner] Refining content with instruction: {instruction}")
        
        return await self.generate_content_plan(
            session_id=session_id,
            instruction=instruction,
            num_cards=num_cards
        )
    
    async def generate_social_post(
        self,
        platform: str,
        caption: str,
        visual_description: str
    ) -> str:
        """
        Generate a platform-specific social media post.
        
        Takes a caption and visual description and expands it into a
        platform-appropriate post format.
        
        Args:
            platform: Target platform (twitter, instagram, linkedin, facebook)
            caption: Short caption to expand
            visual_description: Description of accompanying visual
            
        Returns:
            Platform-formatted post text
        """
        platform_key = (platform or "").strip().lower()
        if platform_key == "x":
            platform_key = "twitter"
        
        # Platform-specific guidance
        guidance_map = {
            "twitter": (
                "Write a single tweet (max 280 chars). Keep it crisp, witty, and high-signal. "
                "Optional 1-2 short hashtags. Avoid excessive emojis. Output ONLY the tweet."
            ),
            "instagram": (
                "Write an Instagram caption in 1-2 short lines. Add 4-8 relevant lowercase hashtags "
                "at the end. Keep it aesthetic, concise, and friendly. Output ONLY the caption."
            ),
            "linkedin": (
                "Write a concise LinkedIn post (2-4 sentences). Professional tone, actionable insight. "
                "Optionally include 1-2 tasteful hashtags. Output ONLY the post text."
            ),
            "facebook": (
                "Write a friendly Facebook post (1-2 sentences). Conversational, approachable. "
                "Avoid overusing hashtags. Output ONLY the post text."
            ),
        }
        
        guidance = guidance_map.get(platform_key, (
            "Write a concise social post (1-2 sentences). "
            "Tailor tone to a general audience. Output ONLY the post text."
        ))
        
        system_instruction = (
            "You are an expert social media copywriter. "
            "Craft platform-appropriate, engaging content. "
            "Do not include disclaimers, prefaces, or metadata."
        )
        
        prompt = (
            f"Platform: {platform_key}\n"
            f"Caption: {caption.strip()}\n"
            f"Visual: {visual_description.strip()}\n\n"
            f"Instructions: {guidance}\n"
        )
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_instruction=system_instruction
            )
            
            cleaned = (response or "").strip()
            
            # Enforce Twitter length limit
            if platform_key == "twitter" and len(cleaned) > 280:
                trimmed = cleaned[:280]
                last_space = trimmed.rfind(" ")
                if last_space > 0:
                    cleaned = trimmed[:last_space].strip()
                else:
                    cleaned = trimmed.strip()
            
            return cleaned
            
        except Exception as e:
            if self.verbose:
                print(f"[SocialPlanner] Error generating social post: {e}")
            raise
