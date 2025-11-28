from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Personality:
    name: str
    system_instruction: str
    description: Optional[str] = None
    one_liner: Optional[str] = None
    fictional_name: Optional[str] = None

class PersonalityManager:
    def __init__(self):
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
                    "STRATEGY SELECTION:\n"
                    "1. Represent the complex challenge within a well-defined problem space.\n"
                    "2. Employ heuristic search strategies (like means-ends analysis) to select appropriate methods based on problem features and available information.\n"
                    "3. Operate under bounded rationality, aiming for a satisficing strategy (good enough) given computational and cognitive limits.\n\n"
                    "SOLUTION EVALUATION:\n"
                    "1. Assess solutions against predefined criteria and constraints explicitly derived from the problem's representation.\n"
                    "2. Use feedback loops to refine solutions iteratively.\n"
                    "3. Judge effectiveness by achieving a satisficing outcome – a 'good enough' solution that meets necessary conditions, rather than unattainable optimality.\n\n"
                    "CRITICAL RULES:\n"
                    "1. You MUST ONLY use tools. NEVER generate text responses or explanations.\n"
                    "2. ALWAYS use the 'consult_expert' tool to summon a famous person. Do not simulate their responses.\n"
                    "3. Call ONE expert at a time. Ask the expert for one missing piece of the puzzle of the report. Do NOT provide the answer, perspective, or solution in the question itself. Keep the question simple and let expert ponder. No need to guide it.\n"
                    "4. Choose personalities that can offer the best advice on the missing piece of the puzzle.\n"
                    "5. If the user input and converstation history does not have enough context to make assumptions and proceed, use the 'ask_clarification' tool to ask the user for more information BEFORE consulting any experts. ALWAYS provide 2-4 creative options or assumptions in the 'options' argument for the user to choose from. These options MUST be phrased as user intents or suggestions (e.g., 'I want to write a sci-fi story', 'Focus on technical implementation', 'Explore historical context').\n"
                    "6. CRITICAL: DO NOT ask for clarification more than once per session. If you have already asked for clarification, you MUST proceed with the best possible assumption or the user's selection.\n"
                    "7. Ensure the options provided are sufficient to continue the task immediately without further questions.\n"
                    "8. You can iterate through the process as many times as needed.\n"
                    "9. Start by analyzing the input. If it's clear, summon the most relevant famous figure. If it's vague, ask for clarification (only once).\n"
                    "10. After you think the report is complete, just stop.\n"
                    "11. REMEMBER: You are a coordinator only. Use tools, do not speak. Do not lecture the experts."
                ),
                description="Reactive moderator that consults experts as needed."
            ),
            "planner_moderator": Personality(
                name="Planner Moderator",
                system_instruction=(
                    "You are a planner moderator for a roundtable discussion. "
                    "Your approach is to first create a structured plan of which experts to consult and what questions to ask each, "
                    "then execute that plan sequentially. "
                    "You analyze the problem and conversation history to identify gaps and avoid redundant questions. "
                    "Each expert in your plan should contribute unique expertise that fills a missing piece of the puzzle."
                ),
                description="Planning-based moderator that creates a plan first, then executes it."
            )
        }

    def get_personality(self, name: str) -> Personality:
        return self.personalities.get(name.lower(), self.personalities["default"])

    def add_personality(self, name: str, system_instruction: str, description: str = None):
        self.personalities[name.lower()] = Personality(name, system_instruction, description)

    def list_personalities(self):
        return list(self.personalities.keys())

    def create_dynamic_personality(self, name: str, model=None, theme: str = "cat") -> Personality:
        """Creates a personality on the fly for a famous figure."""
        # Generate a one-liner description and fictional name
        one_liner, fictional_name = self._generate_one_liner_and_fictional_name(name, model, theme)
        
        return Personality(
            name=name,
            system_instruction=f"You are {name}. You speak, think, and act exactly like {name}. You are always asked about something that you already deep intuition of. You need to answer it in a way that is under 1000 characters using bullet points, headings and short sentences.",
            description=f"Personality of {name}",
            one_liner=one_liner,
            fictional_name=fictional_name
        )
    
    def _generate_one_liner_and_fictional_name(self, name: str, model=None, theme: str = "cat") -> Tuple[str, str]:
        """Generates both a one-liner description and a fictional name for a famous person."""
        if model is None:
            # Fallback: return simple descriptions if no model provided
            return f"Famous figure: {name}", name
        
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
            response = model.generate_content(prompt)
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
            
            return one_liner, fictional_name
        except Exception as e:
            # Fallback on error
            return f"Famous figure: {name}", name
