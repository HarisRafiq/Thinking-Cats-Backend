import asyncio
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modular_agent.database import DatabaseManager

async def verify_archiving():
    # Load env
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    
    db = DatabaseManager()
    await db.connect()
    
    # 1. Create a test session
    print("Creating test session...")
    session_id = await db.create_session({
        "problem": "Test session for archiving",
        "model": "gemini-1.5-pro"
    }, user_id="test_user_id")
    print(f"Created session: {session_id}")
    
    # 2. Verify it appears in get_all_sessions
    sessions = await db.get_all_sessions(user_id="test_user_id")
    found = any(s['session_id'] == session_id for s in sessions)
    print(f"Session found in list: {found}")
    if not found:
        print("ERROR: Session not found in list!")
        return

    # 3. Archive the session
    print("Archiving session...")
    await db.archive_session(session_id)
    
    # 4. Verify it does NOT appear in get_all_sessions
    sessions = await db.get_all_sessions(user_id="test_user_id")
    found = any(s['session_id'] == session_id for s in sessions)
    print(f"Session found in list after archiving: {found}")
    if found:
        print("ERROR: Session still found in list after archiving!")
        return
        
    # 5. Verify it still exists in DB with is_archived=True
    session = await db.get_session(session_id)
    if session and session.get('is_archived') is True:
        print("SUCCESS: Session is archived in DB.")
    else:
        print(f"ERROR: Session state in DB is incorrect: {session}")

    db.close()

if __name__ == "__main__":
    asyncio.run(verify_archiving())
