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
                system_instruction="You are the moderator of a roundtable discussion. You coordinate experts to solve problems by consulting them one at a time.",
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
            system_instruction=f"You are {name}. You speak, think, and act exactly like {name}. Use {name}'s unique perspective, experience, and mannerisms to answer. Your response should be bold, engaging, and humorous but cannot exceed 100 words.",
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
