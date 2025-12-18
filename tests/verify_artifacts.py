
import asyncio
import sys
import os
import json
from pathlib import Path

# Add modular_agent to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from modular_agent.artifact_agent import ArtifactAgent
from modular_agent.database import DatabaseManager

class MockDB:
    async def get_session_context(self, session_id):
        return {
            "problem": "Developing a go-to-market strategy for a new AI coding assistant.",
            "slides": [
                {
                    "type": "agent_response",
                    "sender": "Business Strategist",
                    "content": "We should focus on enterprise developers first, emphasizing security and privacy. A subscription model with tier-based pricing works best."
                },
                {
                    "type": "agent_response",
                    "sender": "Open Source Advocate",
                    "content": "I disagree. We should start with an open-source core to build trust and community. Monetize via cloud hosting and premium plugins."
                },
                {
                    "type": "agent_response",
                    "sender": "Tech Lead",
                    "content": "From a technical perspective, we need a robust plug-in architecture and support for multiple LLMs to avoid lock-in."
                }
            ]
        }
    
    async def get_artifact(self, artifact_id):
        return {"content": "# Project Plan\n\nPhase 1: Initial Research\nPhase 2: MVP Development"}

async def test_suggestions():
    print("\n--- Testing Smarter Suggestions ---")
    db = MockDB()
    agent = ArtifactAgent(db, model_name="gemini-3.0-flash") # type: ignore
    
    suggestions = await agent.get_suggestions("test_session")
    print(f"Suggestions: {json.dumps(suggestions, indent=2)}")
    
    # Verify that we see some "Perspective" or comparison-related suggestions
    actions = suggestions.get("actions", [])
    has_perspective = any("Perspective" in a or "Comparison" in a or "Synthesis" in a for a in actions)
    print(f"Has Perspective-based suggestions: {has_perspective}")
    return has_perspective

async def test_mermaid_generation():
    print("\n--- Testing Mermaid Generation ---")
    db = MockDB()
    agent = ArtifactAgent(db, model_name="gemini-3.0-flash") # type: ignore
    
    print("Generating Mermaid flowchart...")
    content = ""
    async for chunk in agent.generate("test_session", action="Mermaid Flowchart of the product lifecycle"):
        if chunk["type"] == "chunk":
            content += chunk.get("text", "")
        elif chunk["type"] == "complete":
            content = chunk["content"]
            
    print(f"Generated Content Snippet:\n{content[:200]}...")
    is_mermaid = "```mermaid" in content and any(k in content.lower() for k in ["flowchart", "graph ", "sequencediagram"])
    print(f"Contains valid Mermaid block: {is_mermaid}")
    return is_mermaid

async def test_merge_logic():
    print("\n--- Testing Intelligent Merge (Conflict Expected) ---")
    db = MockDB()
    agent = ArtifactAgent(db) # type: ignore
    
    print("Merging new insights...")
    content = ""
    conflict_detected = False
    async for chunk in agent.merge_artifact("test_session", "test_artifact"):
        if chunk["type"] == "chunk":
            content += chunk.get("text", "")
        elif chunk["type"] == "conflict":
            conflict_detected = True
            print(f"Expected conflict detected: {chunk.get('question')}")
        elif chunk["type"] == "complete":
            content = chunk["content"]
            
    if conflict_detected:
        print("✓ Conflict detection working correctly for contradictory inputs.")
        return True
    
    print(f"Merged Content Snippet:\n{content[:200]}...")
    has_original = "Phase 1: Initial Research" in content
    print(f"Preserved original content matching: {has_original}")
    return has_original

async def main():
    s1 = await test_suggestions()
    s2 = await test_mermaid_generation()
    s3 = await test_merge_logic()
    
    print("\n" + "="*30)
    print(f"Final Result: {'PASS' if all([s1, s2, s3]) else 'FAIL'}")
    print("="*30)

if __name__ == "__main__":
    asyncio.run(main())
