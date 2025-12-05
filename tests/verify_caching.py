import asyncio
from unittest.mock import MagicMock, AsyncMock
from modular_agent.personalities import PersonalityManager
from modular_agent.orchestrator import Orchestrator
from modular_agent.llm import GeminiProvider

async def test_caching():
    print("Testing Personality Caching...")
    
    # Mock DB Manager
    mock_db = MagicMock()
    mock_db.get_cached_personality = AsyncMock(return_value=None)
    mock_db.cache_personality = AsyncMock()
    
    # Mock Provider
    mock_provider = MagicMock()
    mock_provider.generate_content_async = AsyncMock()
    mock_provider.generate_content_async.return_value.text = "ONE_LINER: Test One Liner\nFICTIONAL_NAME: Test Name"
    mock_provider.get_token_usage.return_value = {'total_tokens': 10}
    
    # Initialize Manager
    manager = PersonalityManager(db_manager=mock_db)
    
    # Test 1: Cache Miss (Should generate and cache)
    print("\nTest 1: Cache Miss")
    personality, usage = await manager.create_dynamic_personality("Test Expert", provider=mock_provider)
    
    print(f"Generated: {personality.name}")
    assert personality.name == "Test Expert"
    mock_db.get_cached_personality.assert_called_with("Test Expert")
    mock_db.cache_personality.assert_called()
    print("PASS: Cache Miss handled correctly (generated and cached)")
    
    # Test 2: Cache Hit
    print("\nTest 2: Cache Hit")
    mock_db.get_cached_personality.return_value = {
        "name": "test expert",
        "system_instruction": "sys",
        "description": "desc",
        "one_liner": "cached one liner",
        "fictional_name": "cached name"
    }
    mock_db.cache_personality.reset_mock()
    mock_provider.generate_content_async.reset_mock()
    
    personality, usage = await manager.create_dynamic_personality("Test Expert", provider=mock_provider)
    
    print(f"Retrieved: {personality.one_liner}")
    assert personality.one_liner == "cached one liner"
    mock_db.get_cached_personality.assert_called_with("Test Expert")
    mock_provider.generate_content_async.assert_not_called()
    mock_db.cache_personality.assert_not_called()
    print("PASS: Cache Hit handled correctly (no generation, no write)")

def test_orchestrator_model():
    print("\nTesting Orchestrator Model Configuration...")
    orch = Orchestrator(verbose=False)
    
    print(f"One Liner Provider Model: {orch._one_liner_provider.model_name}")
    assert orch._one_liner_provider.model_name == "gemini-2.5-flash"
    print("PASS: Orchestrator uses gemini-2.5-flash for one-liners")

if __name__ == "__main__":
    asyncio.run(test_caching())
    test_orchestrator_model()
