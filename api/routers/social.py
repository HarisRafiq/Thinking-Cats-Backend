from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Literal
import os
import asyncio

# Dependencies
from api.core_dependencies import get_db_manager, get_image_generator
from modular_agent.social_planner import SocialContentPlanner
from modular_agent.image_generator import ImageGenerator
from api.schemas import OptimizationRequest, OptimizationResponse, CardOptimization, PlatformVariant

router = APIRouter(
    prefix="/social",
    tags=["social"],
    responses={404: {"description": "Not found"}},
)

# Story Type Configuration
STORY_TYPES = {
    "narrative": {
        "name": "Narrative Story",
        "description": "A classic story with beginning, middle, and end",
        "prompt_modifier": "Create a compelling narrative story with a clear arc - beginning, middle, and end. Use vivid storytelling to capture the essence of the conversation."
    },
    "hero_journey": {
        "name": "Hero's Journey",
        "description": "Classic transformation arc with challenge and growth",
        "prompt_modifier": "Frame this as a hero's journey: present the challenge or problem, show the struggle and transformation, and conclude with the resolution or new understanding. Focus on growth and overcoming obstacles."
    },
    "before_after": {
        "name": "Before & After",
        "description": "Transformation showing problem to solution",
        "prompt_modifier": "Structure this as a before-and-after transformation story. Clearly show the 'before' state (problem/challenge), the transition or insight gained, and the 'after' state (solution/outcome). Emphasize the contrast."
    },
    "step_by_step": {
        "name": "Step-by-Step Guide",
        "description": "Sequential process broken into actionable steps",
        "prompt_modifier": "Present this as a step-by-step guide or process. Each sentence should represent a clear, actionable step in sequence. Use instructional language and focus on practical implementation."
    },
    "key_insights": {
        "name": "Key Insights",
        "description": "Distilled wisdom and takeaways",
        "prompt_modifier": "Distill the conversation into key insights and takeaways. Each sentence should present one powerful insight, lesson, or piece of wisdom. Focus on clarity and impact over narrative flow."
    },
    "metaphorical": {
        "name": "Metaphorical Journey",
        "description": "Abstract representation using metaphors",
        "prompt_modifier": "Transform this into a metaphorical journey using analogies and symbolic language. Use creative metaphors that make abstract concepts tangible and memorable. Be imaginative and poetic."
    },
    "executive_summary": {
        "name": "Executive Summary",
        "description": "Business-focused, data-driven highlights",
        "prompt_modifier": "Present this as an executive summary with a professional, business-focused tone. Emphasize key points, outcomes, and actionable conclusions. Use clear, authoritative language suitable for decision-makers."
    },
    "teaching_module": {
        "name": "Teaching Module",
        "description": "Educational content structured for learning",
        "prompt_modifier": "Structure this as a teaching module designed for learning. Start with foundational concepts, build complexity progressively, and include examples or applications. Focus on clarity and educational value."
    }
}

StoryType = Literal["narrative", "hero_journey", "before_after", "step_by_step", "key_insights", "metaphorical", "executive_summary", "teaching_module"]

class SocialGenerationRequest(BaseModel):
    session_id: str
    num_cards: int = 4
    story_type: StoryType = "narrative"

class SocialCard(BaseModel):
    url: str
    reading_sentence: str
    visual_sentence: str

class SocialGenerationResponse(BaseModel):
    status: str
    story: Optional[str] = None
    cards: Optional[List[SocialCard]] = None
    error: Optional[str] = None

@router.post("/generate", response_model=SocialGenerationResponse)
async def generate_social_content(
    request: SocialGenerationRequest,
    db_manager = Depends(get_db_manager),
    image_generator: ImageGenerator = Depends(get_image_generator)
):
    """
    Generate social media images based on a story summary of the conversation.
    Two-stage process: 1) Summarize conversation into short story, 2) Create visual grid from story.
    """
    session_id = request.session_id
    num_cards = request.num_cards
    story_type = request.story_type
    
    print(f"[SocialRouter] Starting generation for session {session_id} with story type: {story_type}")
    
 
    try:
        # Step 1: Get session and format conversation
        session = await db_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        
        slides = session.get("slides", [])
        if not slides:
            raise HTTPException(status_code=400, detail="No conversation content to generate from.")
        
        # Format conversation from slides
        conversation = ""
        for slide in slides:
            if slide.get("type") == "user_message":
                if "content" in slide:
                    conversation += f"User: {slide['content']}\n"
                elif "question" in slide and "answer" in slide:
                    conversation += f"Q: {slide['question']}\nA: {slide['answer']}\n"
            elif slide.get("type") == "agent_response":
                sender = slide.get("sender", "Expert")
                content = slide.get("content", "")
                # Include just the first part of response for context
                conversation += f"{sender}: {content}\n"
        
        # Step 2: Generate story summary using LLM
        from modular_agent.llm import GeminiProvider
        llm = GeminiProvider(model_name="gemini-3-flash-preview")
        
        # Get story type configuration
        story_type_config = STORY_TYPES.get(story_type, STORY_TYPES["narrative"])
        story_type_modifier = story_type_config["prompt_modifier"]
        
        story_prompt = f"""You are an expert creative storyteller. Write a compelling short story in EXACTLY {num_cards} sentences that captures the key insights and narrative arc from this conversation.

    Conversation:
    {conversation}

    Story Type: {story_type_config['name']}
    {story_type_modifier}

    Requirements:
    - Reading Narration: Provide a short story in exactly {num_cards} sentences that captures the key insights from this conversation. Follow the story type guidance above.
    - World Narration: Describe the world, setting, characters, and mood where this story happens (a single cohesive paragraph). This should align with the story type.
    - The output must be valid JSON with keys "world_narration" (string) and "reading_narration" (list of {num_cards} strings).
    """
        
        response = llm.generate_content(
            story_prompt, 
            generation_config={"response_mime_type": "application/json"}
        )
        response_text = response.text if response and response.text else None
        
        if not response_text:
            raise HTTPException(status_code=400, detail="Failed to generate story summary.")
            
        import json
        try:
            story_data = json.loads(response_text)
            world_narration = story_data.get("world_narration", "")
            reading_narration = story_data.get("reading_narration", [])

            # Normalize types
            if isinstance(world_narration, list):
                world_narration = "\n".join([str(x) for x in world_narration])
            elif not isinstance(world_narration, str):
                world_narration = str(world_narration)

            # Ensure reading narration has exactly num_cards items
            if not isinstance(reading_narration, list):
                reading_narration = [str(reading_narration)]
            if len(reading_narration) < num_cards:
                reading_narration = reading_narration + [""] * (num_cards - len(reading_narration))
            elif len(reading_narration) > num_cards:
                reading_narration = reading_narration[:num_cards]

            story = "\n".join(reading_narration)
            world_story = world_narration.strip()

        except json.JSONDecodeError:
             raise HTTPException(status_code=500, detail="Failed to parse story generation response.")

        # Step 3: Build image prompt with the story
        master_visual_prompt = f"""
Create a {num_cards}-panel visual narrative for this story.

World Narration (context to apply across all panels):
{world_story}

Reading Narration (one sentence per panel):
{story}

Instructions:
1. Generate exactly {num_cards} images, one per reading sentence, using the shared world context above.
2. Fill each image completely with visuals — no text or numbers.
3. Ensure all images work together as a cohesive visual narrative.
4. Maintain consistent style and theme that matches the world narration.
"""

        # Step 4: Generate Images
        import uuid
        generation_id = str(uuid.uuid4())[:8]
        
        card_urls = await image_generator.generate_cards_from_template(
            prompt=master_visual_prompt,
            num_cards=num_cards,
            session_id=session_id,
            slide_id=f"social_{generation_id}",
            image_size=1024,
            card_padding=15
        )
        
        if not card_urls:
             raise HTTPException(status_code=500, detail="Failed to generate images.")
             
        # Step 5: Format cards with narrative sentences
        cards = [
            {
                "url": card_urls[i],
                "reading_sentence": reading_narration[i],
                "visual_sentence": world_story
            }
            for i in range(len(card_urls))
        ]

        # Step 6: Save to DB
        await db_manager.create_social_generation(session_id, master_visual_prompt, cards, story=story, story_type=story_type)
        
        return SocialGenerationResponse(
            status="success",
            story=story,
            cards=[SocialCard(**c) for c in cards]
        )

    except Exception as e:
        print(f"[SocialRouter] Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{session_id}", response_model=List[dict])
async def get_social_history(
    session_id: str,
    db_manager = Depends(get_db_manager)
):
    """Retrieve history of social generations for a session."""
    try:
        history = await db_manager.get_social_generations(session_id)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/story-types")
async def get_story_types():
    """Get available story types with descriptions."""
    return {
        "story_types": [
            {
                "value": key,
                "name": config["name"],
                "description": config["description"]
            }
            for key, config in STORY_TYPES.items()
        ]
    }

class RefineRequest(BaseModel):
    session_id: str
    instruction: str
    num_cards: int = 4
    story_type: StoryType = "narrative"
    original_generation_id: Optional[str] = None # For context, if we want to retrieve old prompt

@router.post("/refine", response_model=SocialGenerationResponse)
async def refine_social_content(
    request: RefineRequest,
    db_manager = Depends(get_db_manager),
    image_generator: ImageGenerator = Depends(get_image_generator)
):
    """
    Refine existing social content based on user instruction.
    Regenerates story and images with the specified guidance.
    """
    session_id = request.session_id
    instruction = request.instruction
    num_cards = request.num_cards
    story_type = request.story_type
    
    print(f"[SocialRouter] Refining generation for session {session_id} with story type: {story_type}, instruction: {instruction}")
    
    try:
        # Step 1: Get session and format conversation
        session = await db_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        
        slides = session.get("slides", [])
        if not slides:
            raise HTTPException(status_code=400, detail="No conversation content to generate from.")
        
        # Format conversation from slides
        conversation = ""
        for slide in slides:
            if slide.get("type") == "user_message":
                if "content" in slide:
                    conversation += f"User: {slide['content']}\n"
                elif "question" in slide and "answer" in slide:
                    conversation += f"Q: {slide['question']}\nA: {slide['answer']}\n"
            elif slide.get("type") == "agent_response":
                sender = slide.get("sender", "Expert")
                content = slide.get("content", "")
                content_preview = content[:500] + "..." if len(content) > 500 else content
                conversation += f"{sender}: {content_preview}\n"
        
        # Step 2: Generate refined story summary with instruction
        from modular_agent.llm import GeminiProvider
        llm = GeminiProvider(model_name="gemini-3-flash")
        
        # Get story type configuration
        story_type_config = STORY_TYPES.get(story_type, STORY_TYPES["narrative"])
        story_type_modifier = story_type_config["prompt_modifier"]
        
        story_prompt = f"""Based on this conversation, create a compelling narrative consisting of two parts: World Narration and Reading Narration.

    Conversation: {conversation}

    Story Type: {story_type_config['name']}
    {story_type_modifier}

    Refinement instruction: {instruction}

    Requirements:
    - Reading Narration: Provide a refined version of the story text in exactly {num_cards} sentences. Follow the story type guidance and apply the refinement instruction to the tone/focus.
    - World Narration: Describe the world, setting, characters, and mood where this refined story happens (a single cohesive paragraph). This should align with the story type.
    - The output must be valid JSON with keys "world_narration" (string) and "reading_narration" (list of {num_cards} strings).
    """
        
        response = llm.generate_content(
            story_prompt, 
            generation_config={"response_mime_type": "application/json"}
        )
        response_text = response.text if response and response.text else None
        
        if not response_text:
            raise HTTPException(status_code=400, detail="Failed to generate refined story summary.")

        import json
        try:
            story_data = json.loads(response_text)
            world_narration = story_data.get("world_narration", "")
            reading_narration = story_data.get("reading_narration", [])

            # Normalize types
            if isinstance(world_narration, list):
                world_narration = "\n".join([str(x) for x in world_narration])
            elif not isinstance(world_narration, str):
                world_narration = str(world_narration)

            if not isinstance(reading_narration, list):
                reading_narration = [str(reading_narration)]
            if len(reading_narration) < num_cards:
                reading_narration = reading_narration + [""] * (num_cards - len(reading_narration))
            elif len(reading_narration) > num_cards:
                reading_narration = reading_narration[:num_cards]

            story = "\n".join(reading_narration)
            world_story = world_narration.strip()

        except json.JSONDecodeError:
             raise HTTPException(status_code=500, detail="Failed to parse story generation response.")

        # Step 3: Build image prompt with the refined story
        master_visual_prompt = f"""
Create a {num_cards}-panel visual narrative for this refined story.

World Narration (context to apply across all panels):
{world_story}

Reading Narration (one sentence per panel):
{story}

Refinement: {instruction}

Instructions:
1. Generate exactly {num_cards} images, one per reading sentence, using the shared world context above.
2. Select a consistent artistic style that matches the story's mood and the refinement instruction.
3. Each panel should clearly visualize the key action or concept of its corresponding reading sentence.
4. Fill each panel completely with imagery — no text or numbers.
5. Apply the refinement instruction to the visual style and composition.
"""
    
        # Step 4: Generate Images
        import uuid
        generation_id = str(uuid.uuid4())[:8]
        
        card_urls = await image_generator.generate_cards_from_template(
            prompt=master_visual_prompt,
            num_cards=num_cards,
            session_id=session_id,
            slide_id=f"social_refine_{generation_id}",
            image_size=1024,
            card_padding=15
        )
        
        if not card_urls:
             raise HTTPException(status_code=500, detail="Failed to generate refined images.")
             
        # Step 5: Format cards with narrative sentences
        cards = [
            {
                "url": card_urls[i],
                "reading_sentence": reading_narration[i],
                "visual_sentence": world_story
            }
            for i in range(len(card_urls))
        ]

        # Step 6: Save to DB (as new generation)
        await db_manager.create_social_generation(session_id, master_visual_prompt, cards, story=story, story_type=story_type)
        
        return SocialGenerationResponse(
            status="success",
            story=story,
            cards=[SocialCard(**c) for c in cards]
        )

    except Exception as e:
        print(f"[SocialRouter] Refine Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_for_platforms(
    request: OptimizationRequest,
    db_manager = Depends(get_db_manager),
    image_generator: ImageGenerator = Depends(get_image_generator)
):
    """
    Optimizes a social generation for multiple platforms with platform-specific captions,
    hashtags, and image dimensions.
    
    Platforms:
    - Twitter/X: 2:1 aspect ratio, 280 chars, trending hashtags
    - LinkedIn: 1.91:1 aspect ratio, 3000 chars, professional tone
    - Instagram: 1:1 aspect ratio, 2200 chars, visual hashtags
    """
    try:
        # Get the generation
        generation = await db_manager.get_social_generation_by_id(request.generation_id)
        if not generation:
            raise HTTPException(status_code=404, detail="Generation not found")
        
        cards = generation.get("cards", [])
        story = generation.get("story", "")
        session_id = generation.get("session_id", "")
        
        # Use LLM to generate platform-optimized captions
        from modular_agent.llm import GeminiProvider
        llm = GeminiProvider(model_name="gemini-3-flash-preview")
        
        optimized_cards = []
        
        for idx, card in enumerate(cards):
            reading_sentence = card.get("reading_sentence", "")
            visual_sentence = card.get("visual_sentence", "")
            original_url = card.get("url", "")
            
            # Generate platform-specific captions
            platform_prompt = f"""You are a social media expert. Create optimized captions for this content across multiple platforms.

Content: {reading_sentence}
Context: {visual_sentence}
Full Story: {story}

Generate platform-optimized versions:

1. Twitter/X (280 char limit):
   - Punchy, engaging hook
   - 2-3 trending hashtags
   - Conversational tone
   - Character count must be ≤ 280

2. LinkedIn (3000 char limit):
   - Professional, insightful
   - Focus on value and takeaways  
   - 3-5 industry hashtags
   - Can be longer and more detailed

3. Instagram (2200 char limit):
   - Engaging storytelling
   - Visual and emotive language
   - 5-10 relevant hashtags at the end
   - Line breaks for readability

Return JSON with this structure:
{{
  "twitter": {{
    "caption": "...",
    "hashtags": ["...", "..."],
    "char_count": 123
  }},
  "linkedin": {{
    "caption": "...",
    "hashtags": ["...", "..."],
    "char_count": 456
  }},
  "instagram": {{
    "caption": "...",
    "hashtags": ["...", "...", "..."],
    "char_count": 789
  }}
}}
"""
            
            response = llm.generate_content(
                platform_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            import json
            platform_data = json.loads(response.text) if response and response.text else {}
            
            # Create platform variants
            variants = []
            
            # Twitter variant (2:1 - wide)
            twitter_data = platform_data.get("twitter", {})
            variants.append(PlatformVariant(
                platform="twitter",
                caption=twitter_data.get("caption", reading_sentence[:280]),
                hashtags=twitter_data.get("hashtags", []),
                character_count=twitter_data.get("char_count", len(twitter_data.get("caption", ""))),
                image_url=original_url,  # Will generate optimized version
                aspect_ratio="2:1"
            ))
            
            # LinkedIn variant (1.91:1 - wide)
            linkedin_data = platform_data.get("linkedin", {})
            variants.append(PlatformVariant(
                platform="linkedin",
                caption=linkedin_data.get("caption", reading_sentence),
                hashtags=linkedin_data.get("hashtags", []),
                character_count=linkedin_data.get("char_count", len(linkedin_data.get("caption", ""))),
                image_url=original_url,
                aspect_ratio="1.91:1"
            ))
            
            # Instagram variant (1:1 - square)
            instagram_data = platform_data.get("instagram", {})
            variants.append(PlatformVariant(
                platform="instagram",
                caption=instagram_data.get("caption", reading_sentence),
                hashtags=instagram_data.get("hashtags", []),
                character_count=instagram_data.get("char_count", len(instagram_data.get("caption", ""))),
                image_url=original_url,
                aspect_ratio="1:1"
            ))
            
            optimized_cards.append(CardOptimization(
                card_index=idx,
                original_url=original_url,
                reading_sentence=reading_sentence,
                platform_variants=variants
            ))
        
        # Store optimization results
        platform_variants_data = {
            "cards": [card.dict() for card in optimized_cards]
        }
        await db_manager.update_social_generation_platforms(
            request.generation_id,
            platform_variants_data
        )
        
        return OptimizationResponse(
            generation_id=request.generation_id,
            cards=optimized_cards
        )
        
    except Exception as e:
        print(f"[SocialRouter] Optimize Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


