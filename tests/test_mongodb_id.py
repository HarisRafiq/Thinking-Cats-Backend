#!/usr/bin/env python3
"""
Test script to verify MongoDB _id is being used as session_id
"""
import asyncio
import sys
import os

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_agent.database import DatabaseManager
from bson import ObjectId

async def test_mongodb_id_usage():
    print("Testing MongoDB _id as session_id...")
    print("=" * 60)
    
    # 1. Connect to database
    db_manager = DatabaseManager()
    await db_manager.connect()
    print("✓ Connected to MongoDB")
    
    # 2. Create a new session
    session_id = await db_manager.create_session({
        "problem": "Test problem",
        "model": "gemini-2.5-flash"
    })
    print(f"✓ Created session with ID: {session_id}")
    
    # 3. Verify it's a valid ObjectId
    try:
        ObjectId(session_id)
        print(f"✓ Session ID is a valid MongoDB ObjectId")
    except Exception as e:
        print(f"✗ Session ID is NOT a valid ObjectId: {e}")
        return False
    
    # 4. Retrieve the session
    session = await db_manager.get_session(session_id)
    if session:
        print(f"✓ Retrieved session successfully")
        print(f"  - session_id field: {session.get('session_id')}")
        print(f"  - problem: {session.get('problem')}")
        print(f"  - model: {session.get('model')}")
        
        # Verify session_id matches
        if session.get('session_id') == session_id:
            print(f"✓ session_id field matches the ID used for retrieval")
        else:
            print(f"✗ session_id mismatch!")
            return False
    else:
        print(f"✗ Failed to retrieve session")
        return False
    
    # 5. Add a message
    await db_manager.add_message(session_id, {
        "role": "user",
        "content": "Test message"
    })
    print(f"✓ Added message to session")
    
    # 6. Add an event
    await db_manager.add_event(session_id, {
        "type": "test_event",
        "data": "Test event data"
    })
    print(f"✓ Added event to session")
    
    # 7. Retrieve updated session
    updated_session = await db_manager.get_session(session_id)
    if updated_session:
        messages = updated_session.get('messages', [])
        events = updated_session.get('events', [])
        print(f"✓ Retrieved updated session")
        print(f"  - Messages count: {len(messages)}")
        print(f"  - Events count: {len(events)}")
    
    # 8. List all sessions
    all_sessions = await db_manager.get_all_sessions(limit=5)
    print(f"✓ Retrieved {len(all_sessions)} recent sessions")
    
    # Verify all sessions have session_id field
    for sess in all_sessions:
        if 'session_id' not in sess:
            print(f"✗ Session missing session_id field!")
            return False
        if '_id' in sess:
            print(f"✗ Session still has _id field (should be converted to session_id)!")
            return False
    
    print(f"✓ All sessions have session_id field and no _id field")
    
    print("=" * 60)
    print("✅ All tests passed! MongoDB _id is being used correctly.")
    
    # Cleanup
    db_manager.close()
    return True

if __name__ == "__main__":
    result = asyncio.run(test_mongodb_id_usage())
    sys.exit(0 if result else 1)
