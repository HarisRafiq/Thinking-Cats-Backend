"""
Block Generator Agent
Analyzes sessions and generates block-based deliverables with intelligent image placement.
Always generates exactly 4 images for cost optimization.
Production-ready with robust JSON parsing and error recovery.
"""
from typing import Dict, Any, List, Optional, AsyncGenerator
import json
import asyncio
import traceback
from .llm import GeminiProvider
from .database import DatabaseManager
from .image_generator import ImageGenerator
from .utils.json_parser import extract_json_from_response, JSONParseError
from .style_presets import (
    get_writing_style_description,
    get_visual_style_description,
    PRESET_WRITING_STYLES,
    PRESET_VISUAL_STYLES
)


class SuggestionError(Exception):
    """Raised when suggestions cannot be produced or parsed."""


class DeliverableGenerationError(Exception):
    """Raised when deliverable generation fails."""

    def __init__(self, message: str, deliverable_id: Optional[str] = None):
        super().__init__(message)
        self.deliverable_id = deliverable_id


class ImageGenerationError(Exception):
    """Raised when image generation fails for any block."""


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
                    "type": {"type": "string", "enum": ["document", "presentation", "report", "instagram_carousel", "twitter_thread", "linkedin_post"]},
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
            "type": {"type": "string", "enum": ["heading", "text", "image", "quote", "divider", "code", "thread_tweet", "linkedin_post", "instagram_post", "twitter_post"]},
            "order": {"type": "integer"},
            "content": {"type": "object"}
        },
        "required": ["type", "order", "content"]
    }
}


class BlockGenerator:
    """Generates block-based deliverables with optimized image generation."""
    
    DOCUMENT_PROMPT = """You are a world-class content strategist and executive writer.
    
    TASK: Transform the conversation into a comprehensive document with 8-15 blocks.
    
    AVAILABLE BLOCK TYPES:
    - heading: {level: 1-3, text: "Section Title", subtitle: "Optional subtitle"}
    - text: {text: "Full paragraph (markdown supported)", format: "markdown"}
    - list: {items: ["Item 1", "Item 2"], ordered: boolean}
    - statistic: {value: "42%", label: "Growth", trend: "up|down", description: "Context"}
    - callout: {type: "insight|attention|tip", title: "Title", text: "Content"}
    - quote: {text: "Quote text", author: "Name", role: "Title"}
    - table: {headers: ["Col1", "Col2"], rows: [["r1c1", "r1c2"]]}
    - mermaid: {code: "graph TD...", title: "Diagram Title"}
    - image: {prompt: "Detailed image prompt", style: "photorealistic|minimalist|editorial", alt: "Description"}
    - divider: {}
    
    STRUCTURE: Start with heading, use varied blocks, include 2-4 strategic images.
    TONE: Natural, insightful, professional. Avoid AI clichés like "delve" or "tapestry".
    
    OUTPUT: Return ONLY a JSON array of block objects.
    [{"type": "heading", "order": 0, "content": {...}}, ...]
    """
    
    SOCIAL_PROMPT = """You are a social media expert creating engaging posts.
    
    TASK: Transform the conversation into social media content.
    
    OUTPUT: Return ONLY a JSON array of block objects.
    """

    INSTAGRAM_CAROUSEL_PROMPT = """You are an expert Instagram strategist creating a high-converting carousel.

    TASK: Create a 4-slide Instagram Carousel based on the session.
    
    STRUCTURE:
    - Slide 1: Hook (Image + Caption)
    - Slide 2: Value/Insight 1 (Image + Caption)
    - Slide 3: Value/Insight 2 (Image + Caption)
    - Slide 4: Offer/CTA (Image + Caption)
    
    IMPORTANT: You MUST generate exactly 4 image blocks, each followed by an instagram_post block (caption).
    
    AVAILABLE BLOCK TYPES:
    - image: {prompt: "Visual description for the slide", style: "aesthetic|minimalist|bold", alt: "Slide visual"}
    - instagram_post: {text: "Caption for this slide", hashtags: ["tag1"], emoji_line: "✨"}
    
    OUTPUT: Return ONLY a JSON array of block objects.
    [
        {"type": "image", "order": 0, "content": {"prompt": "Visual for hook", "style": "bold"}},
        {"type": "instagram_post", "order": 1, "content": {"text": "Hook caption", "emoji_line": "🔥"}},
        {"type": "image", "order": 2, "content": {"prompt": "Visual for insight 1", "style": "minimalist"}},
        {"type": "instagram_post", "order": 3, "content": {"text": "Insight caption"}}
        ... (continue for 4 slides)
    ]
    """

    TWITTER_THREAD_PROMPT = """You are a Twitter/X ghostwriter creating a viral thread.

    TASK: Create a coherent thread of 5-8 tweets based on the session.
    
    STRUCTURE:
    - Tweet 1: strong hook
    - Tweet 2-N: Body content, 1 idea per tweet
    - Final Tweet: Recap & CTA
    
    AVAILABLE BLOCK TYPES:
    - twitter_post: {text: "Tweet content (max 280 chars)", thread_position: "1/5"}
    - image: {prompt: "Visual for the hook", style: "minimalist"} (Use only 1 image for the first tweet)
    
    OUTPUT: Return ONLY a JSON array of block objects.
    [
        {"type": "image", "order": 0, "content": {"prompt": "Hook visual"}},
        {"type": "twitter_post", "order": 1, "content": {"text": "Hook tweet...", "thread_position": "1/5"}},
        {"type": "twitter_post", "order": 2, "content": {"text": "Point 1...", "thread_position": "2/5"}}
    ]
    """

    LINKEDIN_POST_PROMPT = """You are a LinkedIn thought leader.

    TASK: Create a professional, engagement-optimized LinkedIn post.
    
    STRUCTURE:
    - Hook: Attention grabbing one-liner
    - Body: Spaced out paragraphs, value-dense
    - Takeaway: Bullet points
    - CTA: Question or specific call to action
    
    AVAILABLE BLOCK TYPES:
    - linkedin_post: {text: "Post content", section_type: "body"}
    - image: {prompt: "Professional infographic or concept visualization", style: "professional"} (Include 1 image)
    
    OUTPUT: Return ONLY a JSON array of block objects.
    [
        {"type": "linkedin_post", "order": 0, "content": {"text": "Hook...", "section_type": "hook"}},
        {"type": "image", "order": 1, "content": {"prompt": "Context visual", "style": "professional"}},
        {"type": "linkedin_post", "order": 2, "content": {"text": "Main body...", "section_type": "body"}},
        {"type": "linkedin_post", "order": 3, "content": {"text": "Takeaway...", "section_type": "cta"}}
    ]
    """

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
            raise SuggestionError("Session not found")
        
        # Format conversation
        context = self._format_session_context(session)
        
        # Get suggestions from LLM with structured output config
        prompt = f"Session context:\n{context}\n\nGoal: {session.get('problem', 'Create helpful deliverable')}"
        
        try:
            # Use generation config for JSON output
            system_instruction = """You are an expert content strategist. Analyze the session and suggest 2-4 deliverable options.

Suggestions should include a mix of:
- "document": Comprehensive summaries or guides
- "instagram_carousel": Visual stories (4 slides)
- "twitter_thread": Viral threads (5-8 tweets)
- "linkedin_post": Professional insights

IMPORTANT: Output ONLY valid JSON matching this format:
{"suggestions": [{"title": "...", "type": "document|instagram_carousel|twitter_thread|linkedin_post", "description": "...", "estimated_blocks": 8, "image_count": 4}]}
"""
            response = await self.llm.generate(
                prompt=prompt,
                system_instruction=system_instruction
            )
            
            try:
                result = extract_json_from_response(response, expected_type=dict)
            except JSONParseError as e:
                raise SuggestionError("Unable to parse suggestions JSON") from e

            suggestions = result.get("suggestions", []) if isinstance(result, dict) else []
            
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
            
            if not validated:
                raise SuggestionError("LLM returned no valid suggestions")
            
            # Cache the suggestions
            await self.db.cache_suggestions(session_id, validated)
            
            return validated
            
        except Exception as e:
            print(f"[BlockGenerator] Error in suggest_deliverables: {e}")
            traceback.print_exc()
            if isinstance(e, SuggestionError):
                raise
            raise SuggestionError("Failed to generate suggestions") from e
    
    async def _get_writing_style_description(self, user_id: str, style_id: Optional[str]) -> str:
        """Fetch writing style description (preset or custom)."""
        if not style_id:
            return "Natural, clear writing style"
        
        # Check if it's a preset
        if style_id in PRESET_WRITING_STYLES:
            return get_writing_style_description(style_id)
        
        # Check if it's a custom style
        style = await self.db.get_writing_style(user_id, style_id)
        if style:
            return style.get("description", "Natural, clear writing style")
        
        return "Natural, clear writing style"
    
    async def _get_visual_style_description(self, user_id: str, style_id: Optional[str]) -> str:
        """Fetch visual style description (preset or custom)."""
        if not style_id:
            return "Professional, high-quality visual style"
        
        # Check if it's a preset
        if style_id in PRESET_VISUAL_STYLES:
            return get_visual_style_description(style_id)
        
        # Check if it's a custom style
        style = await self.db.get_visual_style(user_id, style_id)
        if style:
            return style.get("description", "Professional, high-quality visual style")
        
        return "Professional, high-quality visual style"
    
    async def generate_deliverable(
        self,
        user_id: str,
        session_id: str,
        deliverable_type: str,
        title: str,
        instruction: Optional[str] = None,
        deliverable_id: Optional[str] = None,
        writing_style_id: Optional[str] = None,
        visual_style_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate complete deliverable with blocks.
        Yields progress events via SSE with robust error handling.
        """
        deliverable_ref = deliverable_id
        try:
            if not deliverable_ref:
                deliverable_ref = await self.db.create_deliverable(
                    user_id=user_id,
                    deliverable_type=deliverable_type,
                    title=title,
                    session_id=session_id,
                    status="generating"
                )
                yield {
                    "event": "created",
                    "deliverable_id": deliverable_ref,
                    "title": title,
                    "deliverable_type": deliverable_type
                }

            session = await self.db.get_session(session_id)
            if not session:
                raise DeliverableGenerationError("Session not found", deliverable_ref)

            context = self._format_session_context(session)

            writing_style_desc = await self._get_writing_style_description(user_id, writing_style_id)
            visual_style_desc = await self._get_visual_style_description(user_id, visual_style_id)

            yield {"event": "thinking", "message": "Drafting content..."}

            if deliverable_type == "instagram_carousel":
                base_prompt = self.INSTAGRAM_CAROUSEL_PROMPT
            elif deliverable_type == "twitter_thread":
                base_prompt = self.TWITTER_THREAD_PROMPT
            elif deliverable_type == "linkedin_post":
                base_prompt = self.LINKEDIN_POST_PROMPT
            else:
                base_prompt = self.DOCUMENT_PROMPT

            prompt = f"""
            {base_prompt}

            DELIVERABLE DETAILS:
            Title: {title}
            Goal: {instruction or session.get("problem", "Create a helpful guide")}

            STYLE GUIDELINES:
            Writing Style: {writing_style_desc}
            Visual Style for Images: {visual_style_desc}
            IMPORTANT: For all image blocks, use the visual style description above when generating the 'style' field.

            SESSION CONTEXT:
            {context[:5000]}
            """

            response = await self.llm.generate(
                prompt=prompt,
                system_instruction="Output valid JSON array only. No markdown fences."
            )

            try:
                blocks_data = extract_json_from_response(response, expected_type=list)
            except JSONParseError as e:
                raise DeliverableGenerationError("Unable to parse generated blocks", deliverable_ref) from e

            if not blocks_data:
                raise DeliverableGenerationError("LLM returned no blocks", deliverable_ref)

            image_blocks = []

            yield {"event": "generating", "message": "Content generated, saving..."}

            for i, block in enumerate(blocks_data):
                if not isinstance(block, dict):
                    raise DeliverableGenerationError("Block payload is not an object", deliverable_ref)

                block_type = block.get("type", "text")
                content = block.get("content", {})
                order = block.get("order", i)

                if block_type == "text" and isinstance(content, str):
                    content = {"text": content, "format": "markdown"}

                block_id = await self.db.add_block(
                    deliverable_id=deliverable_ref,
                    block_type=block_type,
                    content=content,
                    order=order
                )

                yield {
                    "event": "block_added",
                    "block_id": block_id,
                    "type": block_type,
                    "content": content,
                    "order": order
                }

                if block_type == "image":
                    image_blocks.append({"id": block_id, "content": content})

            if image_blocks:
                if not self.image_service:
                    raise ImageGenerationError("Image generation service unavailable")

                yield {"event": "generating_images", "message": f"Creating {len(image_blocks)} visualizations...", "count": len(image_blocks)}

                async def _gen_update(img_block):
                    prompt_value = img_block["content"].get("prompt")
                    style_value = visual_style_desc if visual_style_id else img_block["content"].get("style", "professional")
                    if not prompt_value:
                        raise ImageGenerationError("Image block missing prompt")

                    full_prompt = f"{prompt_value}, {style_value} style, high quality"
                    try:
                        url = await self.image_service.generate_and_upload_image(
                            prompt=full_prompt,
                            session_id=session_id,
                            slide_id=img_block["id"]
                        )
                    except Exception as exc:
                        raise ImageGenerationError("Image generation failed") from exc

                    if not url:
                        raise ImageGenerationError("Image service returned empty URL")

                    new_content = img_block["content"].copy()
                    new_content["url"] = url
                    await self.db.update_block(deliverable_ref, img_block["id"], new_content)

                    return {
                        "event": "block_updated",
                        "block_id": img_block["id"],
                        "content": new_content
                    }

                tasks = [_gen_update(ib) for ib in image_blocks]
                results = await asyncio.gather(*tasks)

                for res in results:
                    yield res

            await self.db.update_deliverable_status(deliverable_ref, "complete")
            yield {"event": "complete", "deliverable_id": deliverable_ref, "status": "complete"}

        except Exception as e:
            print(f"[BlockGenerator] Fatal error in generate_deliverable: {e}")
            traceback.print_exc()

            if deliverable_ref:
                try:
                    await self.db.update_deliverable_status(deliverable_ref, "failed")
                except Exception:
                    pass

            if isinstance(e, DeliverableGenerationError):
                raise
            raise DeliverableGenerationError(str(e), deliverable_ref) from e
    
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

    async def regenerate_block(
        self,
        deliverable_id: str,
        block_id: str,
        session_id: str,
        instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Regenerate a single block within a deliverable.
        Returns the new block content.
        """
        # Get the existing block
        block = await self.db.get_block(deliverable_id, block_id)
        if not block:
            raise Exception("Block not found")
        
        # Get session context
        session = await self.db.get_session(session_id)
        if not session:
            raise Exception("Session not found")
        
        context = self._format_session_context(session)
        block_type = block.get("type", "text")
        
        # Build regeneration prompt based on block type
        regen_prompt = f"""Regenerate this {block_type} block for a deliverable.

CURRENT BLOCK CONTENT:
{json.dumps(block.get('content', {}), indent=2)}

SESSION CONTEXT:
{context[:3000]}

{f"USER INSTRUCTION: {instruction}" if instruction else "Make it better, more insightful, and more engaging."}

RULES:
- Return ONLY valid JSON with the new content object
- Keep the same block type structure
- Make it more specific and valuable
- If text block, use markdown formatting
"""
        
        # Select appropriate persona based on block type
        persona = "You are an expert content writer. Write with clarity, insight, and a natural human tone."
        if block_type in ["statistic", "chart"]:
            persona = "You are a data analyst. Present information clearly with meaningful context."
        elif block_type in ["twitter_post", "linkedin_post", "instagram_post"]:
            persona = "You are a social media expert. Write engaging, authentic content that resonates."
        
        response = await self.llm.generate(
            prompt=regen_prompt,
            system_instruction=persona + "\n\nWrite naturally. Avoid AI buzzwords. Output ONLY valid JSON.",
            generation_config={"temperature": 0.85}
        )
        
        try:
            new_content = extract_json_from_response(response, expected_type=dict)
        except JSONParseError as e:
            raise DeliverableGenerationError("Unable to parse regenerated block", deliverable_id) from e

        if not isinstance(new_content, dict) or not new_content:
            raise DeliverableGenerationError("Regenerated block is empty", deliverable_id)

        await self.db.update_block(deliverable_id, block_id, new_content)

        return {
            "block_id": block_id,
            "type": block_type,
            "content": new_content,
            "order": block.get("order", 0)
        }

    def export_to_markdown(self, deliverable: Dict[str, Any]) -> str:
        """Export deliverable blocks to markdown format."""
        lines = []
        title = deliverable.get("title", "Untitled")
        lines.append(f"# {title}\n")
        
        for block in deliverable.get("blocks", []):
            block_type = block.get("type", "")
            content = block.get("content", {})
            
            if block_type == "heading":
                level = content.get("level", 1)
                prefix = "#" * min(level + 1, 6)  # +1 since title is h1
                lines.append(f"{prefix} {content.get('text', '')}")
                if content.get("subtitle"):
                    lines.append(f"*{content.get('subtitle')}*")
                lines.append("")
                
            elif block_type == "text":
                lines.append(content.get("text", ""))
                lines.append("")
                
            elif block_type == "list":
                items = content.get("items", [])
                ordered = content.get("ordered", False)
                for i, item in enumerate(items):
                    if ordered:
                        lines.append(f"{i+1}. {item}")
                    else:
                        lines.append(f"- {item}")
                lines.append("")
                
            elif block_type == "statistic":
                lines.append(f"**{content.get('value', '')}** — {content.get('label', '')}")
                if content.get("description"):
                    lines.append(f"_{content.get('description')}_")
                lines.append("")
                
            elif block_type == "callout":
                callout_type = content.get("type", "info").upper()
                title = content.get("title", "")
                text = content.get("text", "")
                lines.append(f"> **{callout_type}**: {title}")
                lines.append(f"> {text}")
                lines.append("")
                
            elif block_type == "quote":
                lines.append(f'> "{content.get("text", "")}"')
                author = content.get("author", "")
                role = content.get("role", "")
                if author:
                    attribution = f"— {author}"
                    if role:
                        attribution += f", {role}"
                    lines.append(f"> {attribution}")
                lines.append("")
                
            elif block_type == "table":
                headers = content.get("headers", [])
                rows = content.get("rows", [])
                if headers:
                    lines.append("| " + " | ".join(headers) + " |")
                    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    for row in rows:
                        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
                lines.append("")
                
            elif block_type == "mermaid":
                lines.append("```mermaid")
                lines.append(content.get("code", ""))
                lines.append("```")
                lines.append("")
                
            elif block_type == "image":
                alt = content.get("alt", "Image")
                url = content.get("url", "")
                if url:
                    lines.append(f"![{alt}]({url})")
                    if content.get("caption"):
                        lines.append(f"*{content.get('caption')}*")
                lines.append("")
                
            elif block_type == "divider":
                lines.append("---")
                lines.append("")
                
            elif block_type in ["thread_tweet", "twitter_post"]:
                position = content.get("thread_position", "")
                if position:
                    lines.append(f"**Tweet {position}**")
                lines.append(content.get("text", ""))
                hashtags = content.get("hashtags", [])
                if hashtags:
                    lines.append(" ".join(f"#{tag}" for tag in hashtags))
                lines.append("")
                
            elif block_type == "linkedin_post":
                section = content.get("section_type", "")
                if section:
                    lines.append(f"**[{section.upper()}]**")
                lines.append(content.get("text", ""))
                lines.append("")
                
            elif block_type == "instagram_post":
                emoji_line = content.get("emoji_line", "")
                if emoji_line:
                    lines.append(emoji_line)
                lines.append(content.get("text", ""))
                hashtags = content.get("hashtags", [])
                if hashtags:
                    lines.append("")
                    lines.append(" ".join(f"#{tag}" for tag in hashtags))
                lines.append("")
        
        return "\n".join(lines)

    def export_to_text(self, deliverable: Dict[str, Any]) -> str:
        """Export deliverable blocks to plain text format."""
        lines = []
        title = deliverable.get("title", "Untitled")
        lines.append(title.upper())
        lines.append("=" * len(title))
        lines.append("")
        
        for block in deliverable.get("blocks", []):
            block_type = block.get("type", "")
            content = block.get("content", {})
            
            if block_type == "heading":
                text = content.get("text", "")
                lines.append(text)
                lines.append("-" * len(text))
                if content.get("subtitle"):
                    lines.append(content.get("subtitle"))
                lines.append("")
                
            elif block_type == "text":
                # Strip markdown formatting for plain text
                text = content.get("text", "")
                # Simple markdown stripping
                text = text.replace("**", "").replace("*", "").replace("`", "")
                lines.append(text)
                lines.append("")
                
            elif block_type == "list":
                items = content.get("items", [])
                for i, item in enumerate(items):
                    lines.append(f"  • {item}")
                lines.append("")
                
            elif block_type == "statistic":
                lines.append(f"{content.get('value', '')} - {content.get('label', '')}")
                if content.get("description"):
                    lines.append(f"  ({content.get('description')})")
                lines.append("")
                
            elif block_type == "callout":
                lines.append(f"[{content.get('type', 'NOTE').upper()}] {content.get('title', '')}")
                lines.append(f"  {content.get('text', '')}")
                lines.append("")
                
            elif block_type == "quote":
                lines.append(f'"{content.get("text", "")}"')
                author = content.get("author", "")
                if author:
                    lines.append(f"  - {author}")
                lines.append("")
                
            elif block_type in ["thread_tweet", "twitter_post", "linkedin_post", "instagram_post"]:
                lines.append(content.get("text", ""))
                lines.append("")
                
            elif block_type == "divider":
                lines.append("-" * 40)
                lines.append("")
        
        return "\n".join(lines)

