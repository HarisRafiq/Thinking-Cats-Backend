"""
Block Generator Agent
Analyzes sessions and generates block-based deliverables with intelligent image placement.
Always generates exactly 4 images for cost optimization.
Production-ready with robust JSON parsing and error recovery.
"""
from typing import Dict, Any, List, Optional, AsyncGenerator
import json
import uuid
import traceback
from .llm import GeminiProvider
from .database import DatabaseManager
from .image_generator import ImageGenerator
from .utils.json_parser import extract_json_from_response, safe_json_loads, JSONParseError


# JSON Schema for structured output
SUGGESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "type": {"type": "string", "enum": ["document", "social", "presentation", "report"]},
                    "description": {"type": "string"},
                    "estimated_blocks": {"type": "integer"},
                    "image_count": {"type": "integer"}
                },
                "required": ["title", "type", "description"]
            }
        }
    },
    "required": ["suggestions"]
}

BLOCKS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": ["heading", "text", "image", "quote", "divider", "code"]},
            "order": {"type": "integer"},
            "content": {"type": "object"}
        },
        "required": ["type", "order", "content"]
    }
}


class BlockGenerator:
    """Generates block-based deliverables with optimized image generation."""
    
    SUGGESTIONS_PROMPT = """You are a senior content strategist at a Fortune 500 company. Analyze this conversation and suggest 3-5 premium deliverables.

DELIVERABLE TYPES:
- document: Executive-level reports, strategic guides, comprehensive plans
- social: High-impact visual narratives for LinkedIn, Instagram, or presentations
- presentation: Board-ready slide decks with data visualization
- report: Data-driven analysis with metrics and insights

For EACH suggestion:
- title: Compelling, specific title (not generic)
- type: Match content to best format
- description: Clear value proposition in 1-2 sentences
- estimated_blocks: 8-12 for comprehensive coverage
- image_count: 4 (our standard)

QUALITY STANDARDS:
- Titles should be specific and compelling (e.g., "Q4 Growth Strategy: 3 Levers for 40% Revenue Increase" not "Business Plan")
- Descriptions should highlight unique insights from the conversation
- Consider what a C-suite executive would find valuable

Output valid JSON only:
{
  "suggestions": [
    {
      "title": "Market Entry Playbook: Southeast Asia Expansion",
      "type": "document",
      "description": "Step-by-step guide with regulatory requirements, partnership strategies, and 18-month timeline based on your discussion",
      "estimated_blocks": 10,
      "image_count": 4
    }
  ]
}"""

    BLOCK_PLANNING_PROMPT = """You are a world-class content designer creating a premium deliverable. Design a compelling block structure.

DESIGN PRINCIPLES:
1. **Visual Hierarchy**: Start strong, build momentum, end with clear takeaways
2. **Data Visualization**: Include charts and diagrams to illustrate key data points
3. **Scanability**: Use varied block types so readers can skim and dive deep
4. **Professional Polish**: Every element should feel intentional and valuable

BLOCK TYPES (use variety for engagement):
- heading: Section titles (content: {level: 1-3, text: "...", subtitle: "optional"})
- text: Rich paragraphs with markdown (content: {text: "...", format: "markdown"})
- list: Bullet or numbered lists (content: {items: ["..."], ordered: boolean})
- statistic: Key metrics with context (content: {value: "85%", label: "Customer Retention", trend: "up", description: "YoY improvement"})
- callout: Highlighted insights (content: {type: "insight"|"action"|"warning"|"tip", title: "...", text: "..."})
- quote: Key quotes or testimonials (content: {text: "...", author: "...", role: "..."})
- table: Comparison or data tables (content: {headers: ["..."], rows: [["..."]], caption: "..."})
- chart: Data visualization (content: {type: "bar"|"line"|"pie"|"area", title: "...", data: [{name: "...", value: N}], xKey: "name", dataKeys: ["value"], caption: "..."})
- mermaid: Diagrams and flowcharts (content: {code: "graph TD\\n  A-->B", title: "Process Flow"})
- image: Strategic visuals (content: {prompt: "detailed description", style: "professional|creative|data-viz", alt: "..."})
- divider: Section breaks (content: {})

CHART DATA FORMAT EXAMPLES:
- Bar/Line chart: {type: "bar", title: "Quarterly Revenue", data: [{"name": "Q1", "value": 120}, {"name": "Q2", "value": 150}], xKey: "name", dataKeys: ["value"]}
- Multi-series: {type: "line", title: "Growth Trends", data: [{"month": "Jan", "revenue": 100, "users": 50}], xKey: "month", dataKeys: ["revenue", "users"]}
- Pie chart: {type: "pie", title: "Market Share", data: [{"name": "Product A", "value": 45}, {"name": "Product B", "value": 30}], xKey: "name", dataKeys: ["value"]}

MERMAID DIAGRAM EXAMPLES:
- Flowchart: graph TD\\n  A[Start] --> B{Decision}\\n  B -->|Yes| C[Action]\\n  B -->|No| D[End]
- Timeline: timeline\\n  title Project Timeline\\n  2024 : Planning\\n  2025 : Development
- Sequence: sequenceDiagram\\n  Alice->>Bob: Hello

STRUCTURE REQUIREMENTS:
- 12-16 total blocks for comprehensive coverage
- Include at least 1 chart block for data visualization
- Include 1 mermaid diagram for process flows or relationships
- Exactly 4 image blocks, strategically placed
- Open with impact (statistic or compelling heading)
- Include at least 1 callout with key insight
- End with actionable takeaways

IMAGE PROMPT QUALITY:
Write detailed, specific image prompts. Bad: "business meeting". Good: "Modern glass boardroom with diverse executive team reviewing growth charts on large display, warm afternoon lighting, professional photography style"

Output valid JSON array only:
[
  {"type": "statistic", "order": 0, "content": {"value": "47%", "label": "Market Opportunity", "trend": "up", "description": "Untapped potential in target segment"}},
  {"type": "heading", "order": 1, "content": {"level": 1, "text": "Strategic Roadmap", "subtitle": "Capturing Market Share in 2025"}},
  {"type": "chart", "order": 2, "content": {"type": "bar", "title": "Revenue by Quarter", "data": [{"name": "Q1", "value": 2.4}, {"name": "Q2", "value": 3.1}], "xKey": "name", "dataKeys": ["value"], "caption": "Values in millions USD"}},
  {"type": "mermaid", "order": 3, "content": {"code": "graph LR\\n  A[Research] --> B[Strategy]\\n  B --> C[Execute]\\n  C --> D[Measure]", "title": "Implementation Process"}}
]"""

    CONTENT_WRITER_PROMPT = """You are an elite business writer crafting content for Fortune 500 executives. 

WRITING STANDARDS:
- Lead with insight, not setup
- Use specific numbers and examples
- Write in active voice
- Be concise but comprehensive
- Include actionable recommendations
- Avoid jargon and buzzwords without substance

FORMAT:
- Use markdown for structure (##, **, -, etc.)
- Break into digestible paragraphs (3-4 sentences max)
- Bold key terms and metrics
- Use bullet points for lists of 3+ items

TONE: Professional, confident, data-informed, actionable"""

    def __init__(self, db_manager: DatabaseManager, llm_provider: GeminiProvider, image_service: ImageGenerator):
        """Initialize block generator."""
        self.db = db_manager
        self.llm = llm_provider
        self.image_service = image_service
    
    async def suggest_deliverables(self, session_id: str, regenerate: bool = False) -> List[Dict[str, Any]]:
        """Generate deliverable suggestions from session with robust error handling."""
        # Check cache first unless regenerate is requested
        if not regenerate:
            cached = await self.db.get_cached_suggestions(session_id)
            if cached:
                return cached
        
        # Get session context
        session = await self.db.get_session(session_id)
        if not session:
            raise Exception("Session not found")
        
        # Format conversation
        context = self._format_session_context(session)
        
        # Get suggestions from LLM with structured output config
        prompt = f"Session context:\n{context}\n\nGoal: {session.get('problem', 'Create helpful deliverable')}"
        
        try:
            # Use generation config for JSON output
            response = await self.llm.generate(
                prompt=prompt,
                system_instruction=self.SUGGESTIONS_PROMPT + "\n\nIMPORTANT: Output ONLY valid JSON. No markdown code fences."
            )
            
            # Use robust JSON parser
            result = extract_json_from_response(response, expected_type=dict, fallback_value={"suggestions": []})
            suggestions = result.get("suggestions", [])
            
            # Validate and normalize suggestions
            validated = []
            for s in suggestions:
                if isinstance(s, dict) and s.get("title") and s.get("type"):
                    validated.append({
                        "title": str(s.get("title", "")),
                        "type": str(s.get("type", "document")),
                        "description": str(s.get("description", "")),
                        "estimated_blocks": int(s.get("estimated_blocks", 8)),
                        "image_count": 4  # Always 4
                    })
            
            # If no valid suggestions, create a default
            if not validated:
                validated = self._get_default_suggestions(session)
            
            # Cache the suggestions
            await self.db.cache_suggestions(session_id, validated)
            
            return validated
            
        except JSONParseError as e:
            print(f"[BlockGenerator] JSON parse error in suggest_deliverables: {e}")
            print(f"[BlockGenerator] Raw response: {e.raw_response[:200]}...")
            print(f"[BlockGenerator] Attempts: {e.attempts}")
            # Return default suggestions on parse error
            return self._get_default_suggestions(session)
        except Exception as e:
            print(f"[BlockGenerator] Error in suggest_deliverables: {e}")
            traceback.print_exc()
            return self._get_default_suggestions(session)
    
    def _get_default_suggestions(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate default suggestions when LLM parsing fails."""
        goal = session.get("problem", "your conversation")[:50]
        return [
            {
                "title": f"Summary Document: {goal}",
                "type": "document",
                "description": "A comprehensive summary of your discussion with key takeaways",
                "estimated_blocks": 8,
                "image_count": 4
            },
            {
                "title": f"Social Media Story: {goal}",
                "type": "social",
                "description": "Visual story with 4 images perfect for social sharing",
                "estimated_blocks": 8,
                "image_count": 4
            }
        ]
    
    async def generate_deliverable(
        self,
        user_id: str,
        session_id: str,
        deliverable_type: str,
        title: str,
        custom_prompt: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate complete deliverable with blocks.
        Yields progress events via SSE with robust error handling.
        """
        deliverable_id = None
        
        try:
            # Create deliverable with 'generating' status
            deliverable_id = await self.db.create_deliverable(
                user_id=user_id,
                deliverable_type=deliverable_type,
                title=title,
                session_id=session_id,
                status="generating"
            )
            
            yield {"event": "created", "deliverable_id": deliverable_id}
            
            # Get session context
            session = await self.db.get_session(session_id)
            if not session:
                raise Exception("Session not found")
            
            context = self._format_session_context(session)
            
            # Plan block structure
            yield {"event": "planning", "message": "Planning content structure..."}
            
            planning_prompt = f"{self.BLOCK_PLANNING_PROMPT}\n\nSession context:\n{context}\n\nDeliverable type: {deliverable_type}\nTitle: {title}"
            if custom_prompt:
                planning_prompt += f"\nUser request: {custom_prompt}"
            
            blocks_json = await self.llm.generate(
                prompt=planning_prompt,
                system_instruction="You are a content structure expert. Output ONLY valid JSON array. No markdown code fences, no explanation."
            )
            
            # Use robust JSON parser
            try:
                blocks = extract_json_from_response(blocks_json, expected_type=list)
            except JSONParseError as e:
                print(f"[BlockGenerator] Failed to parse blocks: {e}")
                print(f"[BlockGenerator] Raw response: {e.raw_response[:300]}...")
                # Generate default blocks structure
                blocks = self._get_default_blocks(deliverable_type, title)
            
            # Handle wrapped response {"blocks": [...]}
            if isinstance(blocks, dict):
                blocks = blocks.get("blocks", [])
            
            # Validate blocks have required fields
            validated_blocks = []
            for i, block in enumerate(blocks):
                if isinstance(block, dict) and block.get("type"):
                    validated_blocks.append({
                        "type": block.get("type", "text"),
                        "order": block.get("order", i),
                        "content": block.get("content", {})
                    })
            
            if not validated_blocks:
                validated_blocks = self._get_default_blocks(deliverable_type, title)
            
            # Separate text and image blocks
            # Note: We now have flexible types that are neither simple text nor image
            image_blocks = [b for b in validated_blocks if b["type"] == "image"]
            non_image_blocks = [b for b in validated_blocks if b["type"] != "image"]
            
            # Ensure exactly 4 image blocks
            while len(image_blocks) < 4:
                image_blocks.append({
                    "type": "image",
                    "order": len(non_image_blocks) + len(image_blocks),
                    "content": {
                        "prompt": f"Visual representation for {title}",
                        "context": "Supporting visual content",
                        "alt": f"Image for {title}"
                    }
                })
            image_blocks = image_blocks[:4]  # Cap at 4
            
            # Generate non-image blocks first
            yield {"event": "generating", "message": "Creating premium content..."}
            
            for block in non_image_blocks:
                try:
                    # Enrich text blocks that are placeholders
                    if block["type"] == "text":
                        text_content = block["content"].get("text", "")
                        if not text_content or "..." in text_content or len(text_content) < 100:
                            content_prompt = f"""Create content for this section of "{title}":

SECTION CONTEXT: {text_content or 'Introduction/overview'}
DELIVERABLE TYPE: {deliverable_type}

CONVERSATION CONTEXT:
{context[:3000]}

Write 2-4 paragraphs of substantive, actionable content. Include specific details, recommendations, or insights from the conversation. Use markdown formatting."""
                            
                            full_text = await self.llm.generate(
                                prompt=content_prompt,
                                system_instruction=self.CONTENT_WRITER_PROMPT
                            )
                            block["content"]["text"] = full_text
                            block["content"]["format"] = "markdown"
                    
                    # Enrich list blocks if items are placeholders
                    elif block["type"] == "list":
                        items = block["content"].get("items", [])
                        if not items or any("..." in str(item) for item in items):
                            list_prompt = f"""Create a list for "{title}":

LIST CONTEXT: {block["content"].get("title", "Key points")}
DELIVERABLE TYPE: {deliverable_type}

CONVERSATION CONTEXT:
{context[:2000]}

Provide 4-7 specific, actionable items. Each item should be 1-2 sentences with concrete details."""
                            
                            list_response = await self.llm.generate(
                                prompt=list_prompt,
                                system_instruction="Output a JSON array of strings. Each string is one list item. Example: [\"First item with detail\", \"Second item\"]"
                            )
                            try:
                                new_items = extract_json_from_response(list_response, expected_type=list, fallback_value=items)
                                if new_items and isinstance(new_items, list):
                                    block["content"]["items"] = new_items
                            except:
                                pass  # Keep original items
                    
                    # Enrich callout blocks
                    elif block["type"] == "callout":
                        callout_text = block["content"].get("text", "")
                        if not callout_text or len(callout_text) < 30:
                            callout_prompt = f"""Create a key insight callout for "{title}":

CALLOUT TYPE: {block["content"].get("type", "insight")}
TITLE: {block["content"].get("title", "Key Insight")}

CONVERSATION CONTEXT:
{context[:1500]}

Write 1-2 sentences highlighting the most important takeaway or action item."""
                            
                            callout_text = await self.llm.generate(
                                prompt=callout_prompt,
                                system_instruction="Write a concise, impactful insight. No preamble, just the insight text."
                            )
                            block["content"]["text"] = callout_text.strip()

                    # Add block to database
                    block_id = await self.db.add_block(
                        deliverable_id=deliverable_id,
                        block_type=block["type"],
                        content=block["content"],
                        order=block["order"]
                    )
                    
                    yield {
                        "event": "block_added",
                        "block_id": block_id,
                        "type": block["type"],
                        "content": block["content"],
                        "order": block["order"]
                    }
                except Exception as block_error:
                    print(f"[BlockGenerator] Error generating block: {block_error}")
                    # Continue with other blocks
            
            # Generate images (exactly 4)
            if image_blocks:
                yield {"event": "generating_images", "message": "Generating 4 images...", "count": len(image_blocks)}
                
                try:
                    # Build combined prompt for all 4 images
                    image_prompts = []
                    for block in image_blocks[:4]:
                        prompt = block["content"].get("prompt", "")
                        img_context = block["content"].get("context", "")
                        combined = f"{img_context}. {prompt}" if img_context else prompt
                        image_prompts.append(combined)
                    
                    # Create master prompt for all images
                    master_prompt = f"Create 4 distinct images for '{title}':\n\n"
                    for i, prompt in enumerate(image_prompts, 1):
                        master_prompt += f"{i}. {prompt}\n"
                    
                    # Generate all 4 images using template method
                    generation_id = str(uuid.uuid4())[:8]
                    card_urls = await self.image_service.generate_cards_from_template(
                        prompt=master_prompt,
                        session_id=user_id,
                        slide_id=generation_id,
                        num_cards=4
                    )
                    
                    if card_urls and len(card_urls) > 0:
                        # Add image blocks to database (as many as we got)
                        for idx, (block, url) in enumerate(zip(image_blocks, card_urls)):
                            content = {
                                "url": url,
                                "caption": block["content"].get("prompt", ""),
                                "alt": block["content"].get("alt", f"Image {idx + 1}"),
                                "width": 512,
                                "height": 512
                            }
                            
                            block_id = await self.db.add_block(
                                deliverable_id=deliverable_id,
                                block_type="image",
                                content=content,
                                order=block["order"]
                            )
                            
                            yield {
                                "event": "block_added",
                                "block_id": block_id,
                                "type": "image",
                                "content": content,
                                "order": block["order"]
                            }
                    else:
                        yield {"event": "warning", "message": "Image generation unavailable, deliverable created without images"}
                        
                except Exception as img_error:
                    print(f"[BlockGenerator] Image generation error: {img_error}")
                    traceback.print_exc()
                    yield {"event": "warning", "message": f"Image generation failed: {str(img_error)[:100]}"}
            
            # Update status to complete
            await self.db.update_deliverable_status(deliverable_id, "complete")
            
            yield {"event": "complete", "deliverable_id": deliverable_id, "status": "complete"}
            
        except Exception as e:
            print(f"[BlockGenerator] Fatal error in generate_deliverable: {e}")
            traceback.print_exc()
            
            # Update status to failed if we have a deliverable_id
            if deliverable_id:
                try:
                    await self.db.update_deliverable_status(deliverable_id, "failed")
                except:
                    pass
            
            yield {"event": "error", "message": str(e), "deliverable_id": deliverable_id}
    
    def _get_default_blocks(self, deliverable_type: str, title: str) -> List[Dict[str, Any]]:
        """Generate default block structure when LLM parsing fails."""
        blocks = [
            {"type": "heading", "order": 0, "content": {"level": 1, "text": title}},
            {"type": "text", "order": 1, "content": {"text": "Introduction and overview.", "format": "markdown"}},
            {"type": "statistic", "order": 2, "content": {"value": "100%", "label": "Compatible", "description": "Works with existing system"}},
            {"type": "image", "order": 3, "content": {"prompt": f"Opening visual for {title}", "context": "Introduction", "alt": "Introduction image"}},
            {"type": "text", "order": 4, "content": {"text": "Key points and insights.", "format": "markdown"}},
            {"type": "list", "order": 5, "content": {"ordered": False, "items": ["Scalable architecture", "Modular design", "Cost effective"]}},
            {"type": "image", "order": 6, "content": {"prompt": f"Key concept visualization for {title}", "context": "Main content", "alt": "Key concept image"}},
            {"type": "callout", "order": 7, "content": {"type": "info", "title": "Note", "text": "This structure is automatically generated as fallback."}},
            {"type": "text", "order": 8, "content": {"text": "Details and implementation.", "format": "markdown"}},
            {"type": "image", "order": 9, "content": {"prompt": f"Supporting visual for {title}", "context": "Details", "alt": "Detail image"}},
            {"type": "text", "order": 10, "content": {"text": "Conclusion and next steps.", "format": "markdown"}},
            {"type": "image", "order": 11, "content": {"prompt": f"Closing visual for {title}", "context": "Conclusion", "alt": "Conclusion image"}},
        ]
        return blocks
    
    def _format_session_context(self, session: Dict[str, Any]) -> str:

        """Format session for LLM context."""
        parts = []
        
        if session.get("problem"):
            parts.append(f"Goal: {session['problem']}")
        
        slides = session.get("slides", [])
        for slide in slides[-10:]:  # Last 10 slides
            if slide.get("user_message"):
                parts.append(f"User: {slide['user_message']}")
            if slide.get("response"):
                parts.append(f"Assistant: {slide['response'][:500]}")  # Truncate
        
        return "\n\n".join(parts)
