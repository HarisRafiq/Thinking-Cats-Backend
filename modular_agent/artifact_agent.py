"""
Artifact Agent Module
Smart agent for creating and updating markdown artifacts from session context.
Provides suggestions for how to generate content, then executes user's choice.
Also provides intelligent editing capabilities with diff-based token efficiency.
"""
from typing import Dict, Any, Optional, AsyncGenerator, Tuple
import json
import re
from rapidfuzz import fuzz, process
import markdown
from .llm import GeminiProvider
from .database import DatabaseManager


class ArtifactAgent:
    """
    Smart agent for artifact generation with suggestion support.
    Analyzes session + existing artifact context to offer options.
    """
    
    SUGGESTIONS_SYSTEM_PROMPT = """Analyze the conversation and suggest 3-5 simple, actionable document types that could be created from it.

Identify the various EXPERT PERSPECTIVES or ROLES present in the session. Suggest documents that synthesize or compare these viewpoints.

OUTPUT FORMAT - Respond with a valid JSON object only:
{"actions": ["Perspective Comparison Matrix", "Executive Synthesis", "Technical Roadmap", "Twitter thread"]}

Be specific to the conversation content. Suggest practical formats like:
- X thread, LinkedIn post, Blog post
- Business plan, Pitch deck, Executive summary
- Todo list, Action plan, Meeting notes
- Email draft, Proposal, Documentation
- Mermaid Flowchart, Sequence Diagram, Gantt Chart

Keep action names SHORT (2-4 words). No emojis, no descriptions.

CRITICAL: Output ONLY valid JSON array of action strings."""

    MERGE_SYSTEM_PROMPT = """You are an intelligent document merger. Your job is to integrate new discussion insights into an existing artifact while maintaining coherence and structural integrity.

ANALYZE for conflicts:
1. Does the new discussion CONTRADICT key points in the original?
2. Does it CHANGE the core argument/conclusion?
3. Are there INCOMPATIBLE perspectives that need user input?

IF CONFLICT DETECTED - respond with JSON:
{
    "conflict": true,
    "question": "Brief question asking user how to resolve the specific conflict",
    "suggested_resolutions": [
        "Option 1: Keep original approach",
        "Option 2: Use new approach",
        "Option 3: Combine both approaches"
    ],
    "conflict_summary": "Describe exactly WHO said WHAT that contradicts the existing document"
}

IF NO CONFLICT - output the merged document directly (not JSON):
- Seamlessly integrate new insights into the existing structure.
- PRESERVE the original flow; do not rewrite sections that don't need changes.
- Focus on EXPANDING existing points or adding NEW sections where appropriate.
- For technical docs or diagrams (like Mermaid), ensure syntax remains valid after integration.
- Maintain the original tone and formatting (Markdown, Mermaid, etc.).

CRITICAL: Either output {"conflict": true, ...} OR output the merged document content directly. Never both."""

    EDITOR_SYSTEM_PROMPT = """You are a surgical document editor. Given a document and an edit instruction, you produce PRECISE, MINIMAL changes.

OUTPUT FORMAT - Respond with a valid JSON object only:
{
    "plan_summary": "1-sentence description of what you're changing",
    "edit_type": "surgical" | "rewrite",
    "edits": [
        {
            "search": "exact text to find (include enough context to be unique)",
            "replace": "the new text to replace it with"
        }
    ]
}

RULES FOR SURGICAL EDITS (preferred when possible):
1. Each edit should be a PRECISE find-and-replace.
2. Include enough surrounding text in "search" to make it UNIQUE.
3. The "search" string MUST exist EXACTLY in the document (including whitespace, punctuation).
4. For Mermaid Diagrams, ensure the search/replace preserves valid Mermaid syntax.
5. Keep edits MINIMAL - only change what's necessary.
6. Order edits from TOP to BOTTOM of the document.

WHEN TO USE "rewrite" INSTEAD:
- If the instruction asks for major restructuring.
- If >25% of the document needs to change.
- If the edit affects the entire flow/narrative.
For rewrites, return a SINGLE edit with search="" (empty) and replace=full new content.

CRITICAL: Output ONLY valid JSON. The search strings must EXACTLY match text in the document."""

    WRITER_SYSTEM_PROMPT = """You are an expert content synthesizer who transforms multi-perspective expert roundtable discussions into polished outputs.

YOUR SOURCE MATERIAL:
Every conversation contains multiple expert perspectives discussing a topic from different angles. Your job is to distill this rich dialogue into the requested format.

SYNTHESIS APPROACH:
1. Identify the core question/topic being addressed
2. Extract key insights, agreements, and disagreements across perspectives
3. Highlight where viewpoints complement or contradict each other
4. Weave different angles into a cohesive narrative
5. Preserve the depth and nuance of the multi-viewpoint discussion

COMMON OUTPUT FORMATS:

**Perspective Comparison Table:**
| Aspect | Perspective 1 | Perspective 2 | Perspective 3 |
Show how different viewpoints address key dimensions

**Consensus Summary:**
What viewpoints aligned on, where they diverged, and implications

**Structured Reports:**
Sections like "Key Insights," "Divergent Views," "Recommendations"
Present contrasting perspectives naturally without attribution

**Social Content:**
- Twitter threads: Different angles as separate tweets, cohesive narrative
- LinkedIn: Synthesize insights showing multiple viewpoints were considered

**Decision Documents:**
Business plans, strategies, action plans that integrate diverse recommendations

**Diagrams:**
Mermaid flowcharts showing relationships between different perspectives
```mermaid syntax with proper structure

**Comparative Analysis:**
Pros/cons, frameworks, or strategic options from different analytical lenses

FORMATTING RULES:
- Use markdown headers, tables, lists, and bold strategically
- Present contrasting views: "Some argue X, while others emphasize Y..."
- Show synthesis: "Combining these perspectives suggests..."
- Highlight tensions: "There's debate between approach A and approach B..."
- Be specific: Use actual concepts and examples from the discussion
- Make it scannable: Break up text, use visual hierarchy

OUTPUT GUIDELINES:
- Lead with a clear title and context about the topic
- Organize content so different perspectives are easy to distinguish
- Synthesize where possible, integrating multiple viewpoints smoothly
- Show the complexity: Don't oversimplify where genuine disagreement exists
- End with key takeaways or recommendations when appropriate
- Default to thorough but scannable formatting

Deliver the content in the format requested, ensuring it captures the multi-dimensional nature of the discussion without attributing ideas to specific sources."""

    def __init__(self, db_manager: DatabaseManager, model_name: str):
        self.db_manager = db_manager
        self.provider = GeminiProvider(model_name=model_name)

    def _strip_outer_code_fence(self, content: str) -> str:
        """Removes a single leading/trailing triple-backtick fence if the whole payload is wrapped."""
        if not content:
            return content

        text = content.strip()
        if not text.startswith("```"):
            return content

        first_newline = text.find("\n")
        if first_newline == -1:
            return content

        opening = text[:first_newline]
        if not opening.startswith("```"):
            return content

        closing_index = text.rfind("```")
        if closing_index <= first_newline:
            return content

        inner = text[first_newline + 1:closing_index]
        return inner.strip()
    
    def _find_fuzzy_match(self, search_text: str, content: str, threshold: int = 85) -> Optional[Tuple[int, int, str]]:
        """
        Finds a fuzzy match for search_text in content.
        Returns (start_pos, end_pos, matched_text) or None if no good match found.
        """
        if not search_text.strip():
            return None
        
        # Try exact match first
        if search_text in content:
            start = content.find(search_text)
            return (start, start + len(search_text), search_text)
        
        # Try fuzzy matching with sliding window
        search_len = len(search_text)
        window_size = min(search_len * 2, len(content))
        best_match = None
        best_score = threshold
        
        # Slide through content with overlapping windows
        step = max(1, search_len // 4)
        for i in range(0, len(content) - window_size + 1, step):
            window = content[i:i + window_size]
            score = fuzz.ratio(search_text, window)
            
            if score > best_score:
                # Try to find the exact boundaries
                # Look for the best substring match within the window
                for j in range(len(window) - search_len + 1):
                    substring = window[j:j + search_len]
                    sub_score = fuzz.ratio(search_text, substring)
                    if sub_score > best_score:
                        best_score = sub_score
                        best_match = (i + j, i + j + search_len, substring)
        
        return best_match
    
    def _validate_markdown(self, content: str) -> Dict[str, Any]:
        """
        Validates markdown structure and returns warnings/issues.
        Returns: {"valid": bool, "warnings": List[str], "errors": List[str]}
        """
        warnings = []
        errors = []
        
        try:
            # Try to parse markdown
            md = markdown.Markdown(extensions=['codehilite', 'tables', 'fenced_code'])
            html = md.convert(content)
            
            # Check for common issues
            # Unclosed code blocks
            code_block_count = content.count('```')
            if code_block_count % 2 != 0:
                errors.append("Unclosed code block detected")
            
            # Unmatched brackets in links/images
            link_pattern = r'\[([^\]]*)\]\(([^\)]*)\)'
            matches = re.findall(link_pattern, content)
            for match in matches:
                if not match[1].strip():
                    warnings.append(f"Empty link URL: {match[0]}")
            
            # Check for broken heading structure (optional warning)
            headings = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
            if len(headings) == 0 and len(content) > 100:
                warnings.append("No headings found in document")
            
        except Exception as e:
            errors.append(f"Markdown parsing error: {str(e)}")
        
        return {
            "valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors
        }
    
    def _format_session_content(self, session_context: Dict[str, Any]) -> str:
        """Formats session into readable content for the LLM, highlighting expert perspectives."""
        parts = []
        
        if session_context.get("problem"):
            parts.append(f"**TOPIC:** {session_context['problem']}\n")
        
        for slide in session_context.get("slides", []):
            slide_type = slide.get("type", "")
            
            if slide_type == "user_message":
                content = slide.get('content', '') or slide.get('answer', '')
                if content:
                    parts.append(f"**USER:** {content}")
            elif slide_type == "agent_response":
                # Use sender/expert name to highlight perspective
                expert = slide.get("sender") or slide.get("expert") or "Expert"
                content = slide.get('content', '')
                parts.append(f"**PERSPECTIVE ({expert}):** {content}")
            elif slide_type == "clarification":
                parts.append(f"**CLARIFICATION REQUEST:** {slide.get('question', '')}")
        
        return "\n\n".join(parts)
    
    async def get_suggestions(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Analyzes session context and returns simple action suggestions.
        Only used for NEW artifact creation. Updates use merge_artifact() directly.
        
        Returns:
            {"actions": ["Twitter thread", "Business plan", ...], "allow_custom": true}
        """
        # Get session context
        session_context = await self.db_manager.get_session_context(session_id)
        if not session_context:
            return {"error": "Session not found", "actions": []}
        
        session_content = self._format_session_content(session_context)
        
        prompt = f"""Analyze this conversation and suggest 3-5 document types that would be useful.

## Session Content
{session_content[:2000]}

What documents could be created from this conversation?"""
        
        try:
            response = await self.provider.generate(
                prompt=prompt,
                system_instruction=self.SUGGESTIONS_SYSTEM_PROMPT
            )
            
            # Parse JSON response
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            result = json.loads(response_text)
            result["allow_custom"] = True
            
            return result
            
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse LLM response: {str(e)}",
                "actions": ["Summary", "Todo list", "Notes"],  # Fallback actions
                "allow_custom": True
            }
        except Exception as e:
            return {
                "error": f"Suggestion generation failed: {str(e)}",
                "actions": ["Summary", "Todo list", "Notes"],
                "allow_custom": True
            }
    
    async def generate(
        self,
        session_id: str,
        action: Optional[str] = None,
        custom_prompt: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generates NEW artifact content based on action or custom prompt.
        For updating existing artifacts, use merge_artifact() instead.
        
        Args:
            session_id: Session to generate from
            action: Simple action string like "Twitter thread", "Business plan"
            custom_prompt: User's custom instruction for what to create
        
        Yields:
            - {"type": "start", "message": str}
            - {"type": "chunk", "text": str}
            - {"type": "complete", "content": str}
            - {"type": "error", "message": str}
        """
        session_context = await self.db_manager.get_session_context(session_id)
        if not session_context:
            yield {"type": "error", "message": "Session not found"}
            return
        
        session_content = self._format_session_content(session_context)
        
        # Determine what to create
        if custom_prompt:
            # User provided custom instructions
            what_to_create = custom_prompt
            display_name = "custom document"
        elif action:
            # User selected a suggested action
            what_to_create = action
            display_name = action.lower()
        else:
            yield {"type": "error", "message": "No action or custom prompt provided"}
            return
        
        prompt = f"""Create content based on the user's request:

**What to Create:** {what_to_create}

## Source Material (Conversation)
{session_content}

Transform this conversation into the requested format."""
        
        yield {"type": "start", "message": f"Creating {display_name}..."}
        
        try:
            full_content = ""
            async for chunk in self.provider.generate_stream(
                prompt=prompt,
                system_instruction=self.WRITER_SYSTEM_PROMPT
            ):
                if chunk.get("text"):
                    full_content += chunk["text"]
                    yield {"type": "chunk", "text": chunk["text"]}
            
            sanitized_content = self._strip_outer_code_fence(full_content)
            
            # Ensure specialized formats have their blocks if missing after stripping
            is_mermaid_requested = any(k in display_name.lower() for k in ["mermaid", "flowchart", "diagram"])
            is_json_requested = any(k in display_name.lower() for k in ["json", "structured"])
            
            if is_mermaid_requested and not sanitized_content.strip().startswith("```mermaid"):
                if any(k in sanitized_content.lower() for k in ["graph ", "flowchart ", "sequencediagram"]):
                    sanitized_content = f"```mermaid\n{sanitized_content.strip()}\n```"
            elif is_json_requested and not sanitized_content.strip().startswith("```json"):
                stripped = sanitized_content.strip()
                if stripped.startswith("{") or stripped.startswith("["):
                    sanitized_content = f"```json\n{stripped}\n```"

            yield {
                "type": "complete",
                "content": sanitized_content
            }
            
        except Exception as e:
            yield {"type": "error", "message": f"Generation failed: {str(e)}"}

    async def merge_artifact(
        self,
        session_id: str,
        artifact_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Automatically merges session content into an existing artifact.
        Detects conflicts and asks for clarification if needed.
        
        Args:
            session_id: Session with new content to merge
            artifact_id: Existing artifact to update
        
        Yields:
            - {"type": "start", "message": str}
            - {"type": "chunk", "text": str} - Streaming merged content
            - {"type": "conflict", "question": str} - Conflict detected, need user input
            - {"type": "complete", "content": str}
            - {"type": "error", "message": str}
        """
        session_context = await self.db_manager.get_session_context(session_id)
        if not session_context:
            yield {"type": "error", "message": "Session not found"}
            return
        
        artifact = await self.db_manager.get_artifact(artifact_id)
        if not artifact:
            yield {"type": "error", "message": "Artifact not found"}
            return
        
        existing_content = artifact.get("content", "")
        if not existing_content.strip():
            yield {"type": "error", "message": "Artifact has no content to merge into"}
            return
        
        session_content = self._format_session_content(session_context)
        
        prompt = f"""Merge the new discussion insights into this existing artifact.

## Existing Artifact
{existing_content}

## New Discussion to Integrate
{session_content}

Analyze for conflicts, then either ask a clarifying question OR output the merged document."""
        
        yield {"type": "start", "message": "Analyzing and merging..."}
        
        try:
            full_response = ""
            async for chunk in self.provider.generate_stream(
                prompt=prompt,
                system_instruction=self.MERGE_SYSTEM_PROMPT
            ):
                if chunk.get("text"):
                    full_response += chunk["text"]
                    # Don't stream yet - we need to check if it's a conflict
            
            # Check if response is a conflict JSON
            response_sanitized = self._strip_outer_code_fence(full_response).strip()
            if response_sanitized.startswith("{"):
                try:
                    conflict_data = json.loads(response_sanitized)
                    if conflict_data.get("conflict"):
                        yield {
                            "type": "conflict",
                            "question": conflict_data.get("question", "How would you like to resolve this conflict?"),
                            "suggested_resolutions": conflict_data.get("suggested_resolutions", []),
                            "conflict_summary": conflict_data.get("conflict_summary", ""),
                            "artifact_id": artifact_id,
                            "session_id": session_id
                        }
                        return
                except json.JSONDecodeError:
                    pass  # Not valid JSON, treat as content
            
            # No conflict - yield the merged content
            sanitized_content = self._strip_outer_code_fence(full_response)
            yield {"type": "chunk", "text": sanitized_content}
            yield {
                "type": "complete",
                "content": sanitized_content
            }
            
        except Exception as e:
            yield {"type": "error", "message": f"Merge failed: {str(e)}"}

    async def resolve_conflict(
        self,
        session_id: str,
        artifact_id: str,
        resolution: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Resolves a merge conflict with user's clarification and completes the merge.
        
        Args:
            session_id: Session with content to merge
            artifact_id: Artifact being updated
            resolution: User's answer/clarification for how to resolve the conflict
        
        Yields:
            - {"type": "start", "message": str}
            - {"type": "chunk", "text": str}
            - {"type": "complete", "content": str}
            - {"type": "error", "message": str}
        """
        session_context = await self.db_manager.get_session_context(session_id)
        if not session_context:
            yield {"type": "error", "message": "Session not found"}
            return
        
        artifact = await self.db_manager.get_artifact(artifact_id)
        if not artifact:
            yield {"type": "error", "message": "Artifact not found"}
            return
        
        existing_content = artifact.get("content", "")
        session_content = self._format_session_content(session_context)
        
        prompt = f"""Merge the new discussion into this artifact using the user's guidance.

## Existing Artifact
{existing_content}

## New Discussion to Integrate
{session_content}

## User's Resolution Guidance
{resolution}

Now merge the content following the user's guidance. Output ONLY the merged document."""
        
        yield {"type": "start", "message": "Completing merge..."}
        
        try:
            full_content = ""
            async for chunk in self.provider.generate_stream(
                prompt=prompt,
                system_instruction=self.WRITER_SYSTEM_PROMPT
            ):
                if chunk.get("text"):
                    full_content += chunk["text"]
                    yield {"type": "chunk", "text": chunk["text"]}
            
            sanitized_content = self._strip_outer_code_fence(full_content)
            yield {
                "type": "complete",
                "content": sanitized_content
            }
            
        except Exception as e:
            yield {"type": "error", "message": f"Resolution failed: {str(e)}"}

    async def edit_artifact(
        self,
        artifact_id: str,
        instruction: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Edits an artifact based on a natural language instruction.
        Uses surgical diff-based edits when possible to save tokens.
        
        Args:
            artifact_id: The artifact to edit
            instruction: Natural language edit instruction (e.g., "Make it more concise")
        
        Yields:
            - {"type": "thinking", "message": str} - Planning phase
            - {"type": "edit", "index": int, "search": str, "replace": str} - Individual edits
            - {"type": "preview", "content": str, "plan_summary": str} - Full preview
            - {"type": "error", "message": str}
        """
        # Get the artifact
        artifact = await self.db_manager.get_artifact(artifact_id)
        if not artifact:
            yield {"type": "error", "message": "Artifact not found"}
            return
        
        current_content = artifact.get("content", "")
        if not current_content.strip():
            yield {"type": "error", "message": "Artifact has no content to edit"}
            return
        
        yield {"type": "thinking", "message": f"Understanding your request: \"{instruction}\"..."}
        
        # Build prompt for the editor
        prompt = f"""Edit this document based on the instruction.

## Current Document
{current_content}

## Edit Instruction
{instruction}

Analyze the instruction and produce the minimal set of changes needed. Prefer surgical edits over full rewrites."""
        
        try:
            # Get edit plan (non-streaming for speed)
            response = await self.provider.generate(
                prompt=prompt,
                system_instruction=self.EDITOR_SYSTEM_PROMPT
            )
            
            # Parse JSON response
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            edit_plan = json.loads(response_text)
            
            plan_summary = edit_plan.get("plan_summary", "Applying edits")
            edit_type = edit_plan.get("edit_type", "surgical")
            edits = edit_plan.get("edits", [])
            
            if not edits:
                yield {"type": "error", "message": "No edits generated"}
                return
            
            yield {"type": "thinking", "message": plan_summary}
            
            # Apply edits to generate preview
            new_content = current_content
            applied_edits = []
            skipped_edits = []
            
            for idx, edit in enumerate(edits):
                search = edit.get("search", "")
                replace = edit.get("replace", "")
                
                if edit_type == "rewrite" and search == "":
                    # Full rewrite
                    new_content = replace
                    yield {
                        "type": "edit",
                        "index": idx,
                        "search": "(full document)",
                        "replace": f"(rewritten - {len(replace)} chars)",
                        "is_rewrite": True
                    }
                    applied_edits.append(edit)
                elif search:
                    # Try exact match first
                    if search in new_content:
                        new_content = new_content.replace(search, replace, 1)
                        yield {
                            "type": "edit",
                            "index": idx,
                            "search": search[:100] + ("..." if len(search) > 100 else ""),
                            "replace": replace[:100] + ("..." if len(replace) > 100 else ""),
                            "is_rewrite": False,
                            "match_type": "exact"
                        }
                        applied_edits.append(edit)
                    else:
                        # Try fuzzy matching
                        fuzzy_match = self._find_fuzzy_match(search, new_content, threshold=80)
                        if fuzzy_match:
                            start, end, matched = fuzzy_match
                            # Replace the matched text
                            new_content = new_content[:start] + replace + new_content[end:]
                            yield {
                                "type": "edit",
                                "index": idx,
                                "search": search[:100] + ("..." if len(search) > 100 else ""),
                                "replace": replace[:100] + ("..." if len(replace) > 100 else ""),
                                "is_rewrite": False,
                                "match_type": "fuzzy",
                                "matched_text": matched[:100] + ("..." if len(matched) > 100 else "")
                            }
                            applied_edits.append(edit)
                        else:
                            # No match found - try partial matching strategies
                            # Strategy 1: Try with normalized whitespace
                            normalized_search = re.sub(r'\s+', ' ', search.strip())
                            normalized_content = re.sub(r'\s+', ' ', new_content)
                            if normalized_search in normalized_content:
                                # Find position in original content
                                pos = normalized_content.find(normalized_search)
                                # Approximate position in original
                                original_pos = new_content.find(normalized_search[:20])
                                if original_pos != -1:
                                    # Try to find the full match around this position
                                    window = new_content[max(0, original_pos-50):original_pos+len(normalized_search)+50]
                                    window_normalized = re.sub(r'\s+', ' ', window)
                                    if normalized_search in window_normalized:
                                        window_pos = window_normalized.find(normalized_search)
                                        actual_start = max(0, original_pos-50) + window_pos
                                        actual_end = actual_start + len(normalized_search)
                                        new_content = new_content[:actual_start] + replace + new_content[actual_end:]
                                        yield {
                                            "type": "edit",
                                            "index": idx,
                                            "search": search[:100] + ("..." if len(search) > 100 else ""),
                                            "replace": replace[:100] + ("..." if len(replace) > 100 else ""),
                                            "is_rewrite": False,
                                            "match_type": "normalized"
                                        }
                                        applied_edits.append(edit)
                                        continue
                            
                            # All strategies failed
                            skipped_edits.append({
                                "index": idx,
                                "search": search[:100],
                                "reason": "Could not find match (exact, fuzzy, or normalized)"
                            })
                            yield {
                                "type": "edit_skipped",
                                "index": idx,
                                "reason": f"Could not find: {search[:50]}..."
                            }
                else:
                    skipped_edits.append({
                        "index": idx,
                        "reason": "Empty search string"
                    })
                    yield {
                        "type": "edit_skipped",
                        "index": idx,
                        "reason": "Empty search string"
                    }
            
            if not applied_edits:
                yield {"type": "error", "message": "No edits could be applied - search strings not found"}
                return
            
            # Validate the edited content
            validation = self._validate_markdown(new_content)
            
            # Yield the preview with validation info
            yield {
                "type": "preview",
                "content": new_content,
                "plan_summary": plan_summary,
                "edit_count": len(applied_edits),
                "edit_type": edit_type,
                "validation": validation,
                "skipped_count": len(skipped_edits)
            }
            
        except json.JSONDecodeError as e:
            yield {"type": "error", "message": f"Failed to parse edit response: {str(e)}"}
        except Exception as e:
            yield {"type": "error", "message": f"Edit failed: {str(e)}"}

