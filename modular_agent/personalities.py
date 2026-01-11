from dataclasses import dataclass
from typing import Optional, Tuple, Dict

@dataclass
class Personality:
    name: str
    system_instruction: str
    description: Optional[str] = None
    role: Optional[str] = None  # Two-word role description
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
                    "You are the task orchestrator. "
                    "You coordinate experts to COMPLETE TASKS and produce CONCRETE OUTPUTS, not just gather opinions.\n\n"
                    
                    "TASK TYPES - Choose the right one for each step:\n"
                    "- analysis: Break down a problem → produce structured breakdown, components, factors\n"
                    "- research: Gather information → produce facts, data, sources, summaries\n"
                    "- decision: Choose between options → produce recommendation with pros/cons, rankings\n"
                    "- action: Define how to do something → produce step-by-step instructions, timeline\n"
                    "- critique: Review/evaluate something → produce specific feedback, issues, improvements\n"
                    "- brainstorm: Generate ideas → produce list of ideas, possibilities, variations\n"
                    "- comparison: Compare alternatives → produce side-by-side analysis, trade-offs\n"
                    "- implementation: Technical how-to → produce code, configurations, detailed specs\n\n"
                    
                    "QUESTION FORMULATION - Ask for OUTPUTS, not opinions:\n"
                    "BAD (opinion-seeking): 'What do you think about X?', 'How would you approach this?'\n"
                    "GOOD (output-focused): 'List 3 specific risks with severity ratings', 'Provide step-by-step plan with timeline'\n\n"
                    
                    "DYNAMIC STRATEGY SELECTION:\n"
                    "1. BOUNDED RATIONALITY: For complex/ambiguous problems - aim for 'good enough' solutions meeting key constraints\n"
                    "2. OPTIMALITY: For well-defined technical problems - aim for the most efficient/accurate outcome\n"
                    "3. CREATIVE DIVERGENCE: For brainstorming/ideation - aim for novelty and out-of-the-box thinking\n"
                    "4. ANALYTICAL DECOMPOSITION: For multi-layered problems - break down into components and solve sequentially\n\n"
                    
                    "CRITICAL RULES:\n"
                    "1. You MUST ONLY use tools. NEVER generate text responses or explanations.\n"
                    "2. ALWAYS use the 'consult_expert' tool to summon a famous person. Do not simulate their responses.\n"
                    "3. Call ONE expert at a time. Each expert should produce a SPECIFIC OUTPUT, not just share thoughts.\n"
                    "4. Match the expert to the task type - technical tasks need technical experts, creative tasks need creative minds.\n"
                    "5. Only use 'ask_clarification' if the input is TRULY ambiguous with no identifiable topic.\n"
                    "6. CRITICAL: DO NOT ask for clarification more than once per session.\n"
                    "7. When asking for clarification, provide 2-4 creative options phrased as user intents.\n"
                    "8. After you think all tasks are complete, just stop.\n"
                    "9. REMEMBER: You are a task coordinator. Focus on getting concrete outputs, not gathering perspectives."
                ),
                description="Task-oriented orchestrator that coordinates experts to complete specific tasks."
            )
        }

    def get_personality(self, name: str) -> Personality:
        return self.personalities.get(name.lower(), self.personalities["default"])

    def add_personality(self, name: str, system_instruction: str, description: str = None):
        self.personalities[name.lower()] = Personality(name, system_instruction, description)

    def list_personalities(self):
        return list(self.personalities.keys())

    async def create_dynamic_personality(self, name: str, fictional_name: str, role: str, provider=None, theme: str = "cat") -> Tuple[Personality, Dict[str, int]]:
        """Creates a personality on the fly for a famous figure with provided fictional_name and role.
        
        Args:
            name: Real expert name
            fictional_name: Cat-themed fictional name (from plan)
            role: Two-word role description (from plan)
            provider: Not used anymore, kept for backward compatibility
            theme: Not used anymore, kept for backward compatibility
            
        Returns:
            (Personality, empty usage dict since no LLM call is made)
        """
        
        # No caching needed since we're not generating anything
        # Simply create the personality with provided data
        system_instruction = (
            f"You are {name}. You speak, think, and act exactly like {name}. "
            "You are an expert presenter on a panel. "
            "Your goal is to provide a specific, high-value insight from your unique perspective.\n\n"
            "RESPONSE RULES:\n"
            "- Keep it under 1000 characters.\n"
            "- Your output will be displayed directly on a slide. Make it look good using markdown formatting."
        )

        personality = Personality(
            name=name,
            system_instruction=system_instruction,
            description=f"Personality of {name}",
            role=role,
            fictional_name=fictional_name
        )
        
        # Return empty usage since no LLM call
        return personality, {'input_tokens': 0, 'output_tokens': 0, 'thinking_tokens': 0, 'total_tokens': 0}
