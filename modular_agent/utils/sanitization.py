from typing import Dict

def sanitize_response(text: str, active_experts: Dict[str, str], real_name: str = None, fictional_name: str = None) -> str:
    """
    Replaces real names with fictional names in the text.
    If real_name/fictional_name are provided, it prioritizes them.
    It also checks active_experts for other replacements.
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
    for r_name, f_name in active_experts.items():
        if r_name != real_name: # Already handled
            sanitized = replace_variants(sanitized, r_name, f_name)
            
    return sanitized
