import asyncio
import os
import sys
import time
from typing import Dict, Any

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_agent.orchestrator import Orchestrator
from modular_agent.database import DatabaseManager

async def test_context_restoration():
    print("Starting Context Restoration Test...")
    
    # 1. Setup
    db_manager = DatabaseManager()
    await db_manager.connect()
    
    # Create new session in MongoDB to get the _id
    problem = "What is 2 + 2?"
    session_id = await db_manager.create_session({
        "problem": problem
    })
    print(f"Created session ID: {session_id}")
    
    # 2. Create initial orchestrator and simulate a conversation
    print("\n--- Phase 1: Initial Conversation ---")
    orchestrator1 = Orchestrator(
        session_id=session_id,
        db_manager=db_manager,
        verbose=True
    )
    
    # Simulate solving a problem
    # Note: We must manually add the initial message to DB as main.py does
    await db_manager.add_message(session_id, {"role": "user", "content": problem})
    await orchestrator1.process(problem)

    
    # Simulate a user follow-up
    user_msg = "Are you sure?"
    # Manually add timestamped message to DB as main.py would
    await db_manager.add_message(session_id, {
        "role": "user", 
        "content": user_msg,
        "timestamp": time.time()
    })
    
    # Wait a moment for async operations to complete
    await asyncio.sleep(2)
    
    # 3. Check what's in the database
    print("\n--- Inspecting Database ---")
    session_data = await db_manager.get_session(session_id)
    if session_data:
        print(f"Session found in DB")
        print(f"  Problem: {session_data.get('problem')}")
        print(f"  Events count: {len(session_data.get('events', []))}")
        print(f"  Messages count: {len(session_data.get('messages', []))}")
        
        # Print first few events
        for i, event in enumerate(session_data.get('events', [])[:5]):
            print(f"  Event {i}: {event.get('type')}")
            
        # Print messages
        for i, msg in enumerate(session_data.get('messages', [])):
            print(f"  Message {i}: {msg.get('role')} - {msg.get('content', '')[:50]}")
    else:
        print("ERROR: Session not found in database!")
    
    # 4. Simulate Restart
    print("\n--- Phase 2: Simulating Restart ---")
    # Create a new orchestrator instance for the same session
    orchestrator2 = Orchestrator(
        session_id=session_id,
        db_manager=db_manager,
        verbose=True
    )
    
    # Verify context is empty initially
    print(f"Initial context length (should be 0): {len(orchestrator2.agent.context)}")
    assert len(orchestrator2.agent.context) == 0
    
    # 5. Load History
    print("\n--- Phase 3: Loading History ---")
    await orchestrator2.load_session_history()
    
    # 6. Verify Context
    print(f"Restored context length: {len(orchestrator2.agent.context)}")
    
    context_messages = orchestrator2.agent.context.messages
    for i, msg in enumerate(context_messages):
        role = msg.get('role')
        content = msg.get('content')
        fn_call = msg.get('function_call')
        fn_resp = msg.get('function_response')
        
        print(f"[{i}] {role}: {content[:50]}..." if content else f"[{i}] {role}: [Function Call/Response]")
        if fn_call:
            print(f"    Function Call: {fn_call.name}")
        if fn_resp:
            print(f"    Function Response: {fn_resp.name}")

    # Basic assertions
    # We expect at least:
    # 1. User message (Problem)
    # 2. Model message (Orchestrator thought/plan) - likely text
    # 3. User message (Follow up "Are you sure?")
    # 4. Model message (Reply)
    
    # Note: The exact number depends on how many steps the orchestrator took.
    # But we should definitely see the "Problem: ..." message and "Are you sure?" message.
    
    has_problem = any("What is 2 + 2?" in m.get('content', '') for m in context_messages if m.get('role') == 'user')
    has_followup = any("Are you sure?" in m.get('content', '') for m in context_messages if m.get('role') == 'user')
    
    if has_problem:
        print("SUCCESS: Found problem statement in restored context.")
    else:
        print("FAILURE: Problem statement not found in restored context.")
        
    if has_followup:
        print("SUCCESS: Found follow-up message in restored context.")
    else:
        print("FAILURE: Follow-up message not found in restored context.")
        
    if has_problem and has_followup:
        print("\nTEST PASSED: Context restoration verified.")
    else:
        print("\nTEST FAILED: Context missing elements.")

if __name__ == "__main__":
    asyncio.run(test_context_restoration())
