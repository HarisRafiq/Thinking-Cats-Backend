from dataclasses import dataclass
from typing import Optional, Tuple, Dict

@dataclass
class Personality:
    name: str
    system_instruction: str
    description: Optional[str] = None
    one_liner: Optional[str] = None
    fictional_name: Optional[str] = None

class PersonalityManager:
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.personalities = {
            "default": Personality(
                name="Default",
                system_instruction="You are a helpful and capable AI assistant.",
                description="Standard helpful assistant."
            ),
            "moderator": Personality(
                name="Moderator",
                system_instruction=(
                    "You are the moderator of a roundtable discussion. "
                    "You coordinate experts to help user makes better decisions by consulting them one at a time.\n\n"
                    "DYNAMIC STRATEGY SELECTION:\n"
                    "Analyze the user's request and implicitly adopt ONE of the following strategies to guide your expert consultation process:\n"
                    "1. BOUNDED RATIONALITY (Satisficing): Use for complex, ambiguous, or 'wicked' problems where an optimal solution is impossible or too costly. Aim for a 'good enough' solution that meets key constraints. Consult diverse experts to cover different angles.\n"
                    "2. OPTIMALITY (Maximizing): Use for well-defined, technical, or logical problems where a clear 'best' solution exists. Aim for the most efficient, accurate, or high-performance outcome. Consult domain specialists (e.g., mathematicians, engineers).\n"
                    "3. CREATIVE DIVERGENCE: Use for brainstorming, ideation, or artistic tasks. Aim for novelty, variety, and out-of-the-box thinking. Consult creative figures (e.g., artists, writers, visionaries).\n"
                    "4. ANALYTICAL DECOMPOSITION: Use for structural, systemic, or multi-layered problems. Break the problem down into smaller components and solve them sequentially. Consult experts who can handle specific sub-components.\n\n"
                    "SOLUTION EVALUATION:\n"
                    "1. Assess solutions against the criteria defined by your chosen strategy.\n"
                    "2. Use feedback loops to refine solutions iteratively.\n"
                    "3. Ensure the final outcome aligns with the user's intent and the chosen strategy's goal.\n\n"
                    "CRITICAL RULES:\n"
                    "1. You MUST ONLY use tools. NEVER generate text responses or explanations.\n"
                    "2. ALWAYS use the 'consult_expert' tool to summon a famous person. Do not simulate their responses.\n"
                    "3. Call ONE expert at a time. Ask the expert for one missing piece of the puzzle of the report. Do NOT provide the answer, perspective, or solution in the question itself. Keep the question simple and let expert ponder. No need to guide it.\n"
                    "4. Choose personalities that can offer the best advice on the missing piece of the puzzle.\n"
                    "5. If the user input and converstation history does not have enough context to make assumptions and proceed, use the 'ask_clarification' tool to ask the user for more information BEFORE consulting any experts. Check the conversation history carefully. If the user input is a follow-up, use the existing context. ALWAYS provide 2-4 creative options or assumptions in the 'options' argument for the user to choose from. These options MUST be phrased as user intents or suggestions (e.g., 'I want to write a sci-fi story', 'Focus on technical implementation', 'Explore historical context').\n"
                    "6. CRITICAL: DO NOT ask for clarification more than once per session. If you have already asked for clarification, you MUST proceed with the best possible assumption or the user's selection.\n"
                    "7. Ensure the options provided are sufficient to continue the task immediately without further questions.\n"
                    "8. You can iterate through the process as many times as needed.\n"
                    "9. Start by analyzing the input. If it's clear, summon the most relevant famous figure. If it's vague, ask for clarification (only once).\n"
                    "10. After you think the report is complete, just stop.\n"
                    "11. REMEMBER: You are a coordinator only. Use tools, do not speak. Do not lecture the experts."
                ),
                description="Reactive moderator that consults experts as needed."
            )
        }

    def get_personality(self, name: str) -> Personality:
        return self.personalities.get(name.lower(), self.personalities["default"])

    def add_personality(self, name: str, system_instruction: str, description: str = None):
        self.personalities[name.lower()] = Personality(name, system_instruction, description)

    def list_personalities(self):
        return list(self.personalities.keys())

    async def create_dynamic_personality(self, name: str, provider=None, theme: str = "cat") -> Tuple[Personality, Dict[str, int]]:
        """Creates a personality on the fly for a famous figure. Returns (Personality, usage_dict)."""
        
        # Check cache first
        if self.db_manager:
            cached = await self.db_manager.get_cached_personality(name)
            if cached:
                # Regenerate system instruction
                system_instruction = (
                    f"You are {cached['name']}. You speak, think, and act exactly like {cached['name']}. "
                    "You are always asked about something that you already have deep intuition of and could relate to it. "
                    "You need to respond in an engaging personal way in under 1000 characters and format your response in markdown with headings and short sentences."
                )
                
                personality = Personality(
                    name=cached['name'],
                    system_instruction=system_instruction,
                    description=f"Personality of {cached['name']}",
                    one_liner=cached.get('one_liner'), # Use get in case old cache doesn't have it
                    fictional_name=cached.get('fictional_name')
                )
                # Return empty usage for cached hit
                return personality, {'input_tokens': 0, 'output_tokens': 0, 'thinking_tokens': 0, 'total_tokens': 0}

        # Generate a one-liner description and fictional name
        one_liner, fictional_name, usage = await self._generate_one_liner_and_fictional_name(name, provider, theme)
        
        system_instruction = (
            f"You are {name}. You speak, think, and act exactly like {name}. "
            "You are always asked about something that you already have deep intuition of and could relate to it. "
            "You need to respond in an engaging personal way in under 1000 characters and format your response in markdown with headings and short sentences."
        )

        personality = Personality(
            name=name,
            system_instruction=system_instruction,
            description=f"Personality of {name}",
            one_liner=one_liner,
            fictional_name=fictional_name
        )
        
        # Cache the result
        if self.db_manager:
            await self.db_manager.cache_personality(name, {
                "one_liner": personality.one_liner,
                "fictional_name": personality.fictional_name
            })
            
        return personality, usage
    
    async def _generate_one_liner_and_fictional_name(self, name: str, provider=None, theme: str = "cat") -> Tuple[str, str, Dict[str, int]]:
        """Generates both a one-liner description and a fictional name for a famous person."""
        usage = {'input_tokens': 0, 'output_tokens': 0, 'thinking_tokens': 0, 'total_tokens': 0}
        
        if provider is None:
            # Fallback: return simple descriptions if no provider provided
            return f"Famous figure: {name}", name, usage
        
        try:
            prompt = (
        f"For the famous person {name}, provide:\n"
        f"1. A fictional {theme}-themed name inspired by the person’s real name."
        f"Ensure it is clearly humorous, playful and legally safe. But it should be simple for readers to pick up the hint for real person.\n\n"
        f"2. A concise one-liner why this person is best to answer the question. Be witty and funny."
        f"Max 10 words.\n\n"
        f"Format your response EXACTLY as:\n"
        f"ONE_LINER: [the one-liner]\n"
        f"FICTIONAL_NAME: [the fictional name]\n\n"
            )
            
            # Use provider to generate content
            if hasattr(provider, 'generate_content_async'):
                response = await provider.generate_content_async(prompt)
            else:
                response = provider.generate_content(prompt)
            
            # Extract usage
            if hasattr(provider, 'get_token_usage'):
                usage = provider.get_token_usage(response)
            
            text = response.text.strip()
            
            # Parse the response
            one_liner = None
            fictional_name = None
            
            for line in text.split('\n'):
                if line.startswith('ONE_LINER:'):
                    one_liner = line.replace('ONE_LINER:', '').strip().strip('"\'')
                elif line.startswith('FICTIONAL_NAME:'):
                    fictional_name = line.replace('FICTIONAL_NAME:', '').strip().strip('"\'')
            
            # Fallback if parsing failed
            if not one_liner or not fictional_name:
                # Try to extract from the response more flexibly
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                if len(lines) >= 2:
                    one_liner = lines[0].strip('"\'')
                    fictional_name = lines[1].strip('"\'')
                else:
                    one_liner = f"Famous figure: {name}"
                    fictional_name = name
            
            # Ensure we have valid values
            one_liner = one_liner[:100] if one_liner else f"Famous figure: {name}"
            fictional_name = fictional_name if fictional_name else name
            
            return one_liner, fictional_name, usage
        except Exception as e:
            # Fallback on error
            print(f"Error generating dynamic personality: {e}")
            return f"Famous figure: {name}", name, usage
