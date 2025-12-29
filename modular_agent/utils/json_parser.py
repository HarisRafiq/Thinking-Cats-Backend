"""
Robust JSON Parser for LLM Responses
Handles malformed JSON, markdown fences, and partial responses with multiple fallback strategies.
"""
import re
import json
from typing import Any, Optional, List, Dict, Union


class JSONParseError(Exception):
    """Custom exception for JSON parsing failures with context."""
    def __init__(self, message: str, raw_response: str, attempts: List[str]):
        super().__init__(message)
        self.raw_response = raw_response
        self.attempts = attempts


def extract_json_from_response(
    response: str,
    expected_type: type = dict,
    fallback_value: Optional[Any] = None
) -> Any:
    """
    Robustly extract JSON from LLM response with multiple fallback strategies.
    
    Args:
        response: Raw LLM response text
        expected_type: Expected JSON type (dict or list)
        fallback_value: Value to return if all parsing fails
        
    Returns:
        Parsed JSON object or fallback_value
        
    Raises:
        JSONParseError: If parsing fails and no fallback provided
    """
    if not response or not response.strip():
        if fallback_value is not None:
            return fallback_value
        raise JSONParseError("Empty response", response or "", ["empty check"])
    
    attempts = []
    original = response.strip()
    
    # Strategy 1: Direct parse (response is already valid JSON)
    try:
        result = json.loads(original)
        if isinstance(result, expected_type):
            return result
        attempts.append("direct_parse: wrong type")
    except json.JSONDecodeError as e:
        attempts.append(f"direct_parse: {str(e)[:50]}")
    
    # Strategy 2: Strip markdown code fences
    cleaned = _strip_markdown_fences(original)
    if cleaned != original:
        try:
            result = json.loads(cleaned)
            if isinstance(result, expected_type):
                return result
            attempts.append("strip_fences: wrong type")
        except json.JSONDecodeError as e:
            attempts.append(f"strip_fences: {str(e)[:50]}")
    
    # Strategy 3: Extract JSON using bracket matching
    extracted = _extract_json_by_brackets(original, expected_type)
    if extracted:
        try:
            result = json.loads(extracted)
            if isinstance(result, expected_type):
                return result
            attempts.append("bracket_extract: wrong type")
        except json.JSONDecodeError as e:
            attempts.append(f"bracket_extract: {str(e)[:50]}")
    
    # Strategy 4: Regex-based extraction
    extracted = _extract_json_by_regex(original, expected_type)
    if extracted:
        try:
            result = json.loads(extracted)
            if isinstance(result, expected_type):
                return result
            attempts.append("regex_extract: wrong type")
        except json.JSONDecodeError as e:
            attempts.append(f"regex_extract: {str(e)[:50]}")
    
    # Strategy 5: Try to repair common JSON issues
    repaired = _repair_json(cleaned or original)
    if repaired:
        try:
            result = json.loads(repaired)
            if isinstance(result, expected_type):
                return result
            attempts.append("repair: wrong type")
        except json.JSONDecodeError as e:
            attempts.append(f"repair: {str(e)[:50]}")
    
    # Strategy 6: For arrays, try extracting individual objects
    if expected_type == list:
        objects = _extract_json_objects(original)
        if objects:
            return objects
        attempts.append("object_extraction: no objects found")
    
    # All strategies failed
    if fallback_value is not None:
        return fallback_value
    
    raise JSONParseError(
        f"Failed to parse JSON after {len(attempts)} attempts",
        original[:500],
        attempts
    )


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    text = text.strip()
    
    # Handle ```json ... ``` or ``` ... ```
    patterns = [
        r'^```json\s*\n?(.*?)\n?```$',
        r'^```\s*\n?(.*?)\n?```$',
        r'```json\s*\n(.*?)\n```',
        r'```\s*\n(.*?)\n```',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Simple prefix/suffix removal
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    
    if text.endswith('```'):
        text = text[:-3]
    
    return text.strip()


def _extract_json_by_brackets(text: str, expected_type: type) -> Optional[str]:
    """Extract JSON by finding matching brackets."""
    open_bracket = '{' if expected_type == dict else '['
    close_bracket = '}' if expected_type == dict else ']'
    
    # Find first opening bracket
    start = text.find(open_bracket)
    if start == -1:
        return None
    
    # Track bracket depth to find matching close
    depth = 0
    in_string = False
    escape_next = False
    
    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
            
        if char == '\\' and in_string:
            escape_next = True
            continue
            
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
            
        if in_string:
            continue
            
        if char == open_bracket:
            depth += 1
        elif char == close_bracket:
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    
    return None


def _extract_json_by_regex(text: str, expected_type: type) -> Optional[str]:
    """Extract JSON using regex patterns."""
    if expected_type == dict:
        # Match object pattern
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, text, re.DOTALL)
        # Return the longest match (most complete object)
        return max(matches, key=len) if matches else None
    else:
        # Match array pattern
        pattern = r'\[[\s\S]*?\]'
        matches = re.findall(pattern, text, re.DOTALL)
        return max(matches, key=len) if matches else None


def _repair_json(text: str) -> Optional[str]:
    """Attempt to repair common JSON issues."""
    if not text:
        return None
    
    repaired = text
    
    # Fix trailing commas before closing brackets
    repaired = re.sub(r',\s*([\]}])', r'\1', repaired)
    
    # Fix missing quotes around keys (simple cases)
    repaired = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
    
    # Fix single quotes to double quotes (be careful with content)
    # Only do this for obvious key patterns
    repaired = re.sub(r"'([a-zA-Z_][a-zA-Z0-9_]*)'\s*:", r'"\1":', repaired)
    
    # Remove comments (// style)
    repaired = re.sub(r'//[^\n]*\n', '\n', repaired)
    
    # Fix unescaped newlines in strings (dangerous, only simple cases)
    # This is risky so we skip it
    
    return repaired if repaired != text else None


def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Extract multiple JSON objects from text (for array responses)."""
    objects = []
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    
    for match in re.finditer(pattern, text, re.DOTALL):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            continue
    
    return objects


def safe_json_loads(
    text: str,
    default: Any = None,
    expected_type: type = dict
) -> Any:
    """
    Safe JSON loading with fallback - never raises.
    
    Args:
        text: JSON string to parse
        default: Default value if parsing fails
        expected_type: Expected result type
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return extract_json_from_response(text, expected_type, default)
    except JSONParseError:
        return default
