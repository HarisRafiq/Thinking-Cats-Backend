import asyncio
import aiohttp
import sys
import json

async def verify_api():
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        # 1. Start a chat
        print("Starting chat...")
        async with session.post(f"{base_url}/chat", json={"problem": "Why is the sky blue?"}) as resp:
            if resp.status != 200:
                print(f"Failed to start chat: {await resp.text()}")
                return
            data = await resp.json()
            session_id = data['session_id']
            print(f"Chat started with session_id: {session_id}")

        # 2. Listen to stream briefly to ensure events are flowing and saved
        print("Listening to stream...")
        # We'll just connect and disconnect quickly or wait for a few events
        # Since we can't easily consume SSE with simple aiohttp get in a short script without a loop,
        # we'll just check if the endpoint is accessible.
        # Actually, let's try to read a bit.
        try:
            async with session.get(f"{base_url}/stream/{session_id}") as resp:
                print(f"Stream status: {resp.status}")
                # Read a few chunks
                async for line in resp.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data:"):
                        print(f"Received event: {line[:50]}...")
                        break # Received at least one event
        except Exception as e:
            print(f"Stream error (expected if we cut it short): {e}")

        # 3. Check if session is listed
        print("Checking session list...")
        async with session.get(f"{base_url}/sessions") as resp:
            sessions = await resp.json()
            found = any(s['session_id'] == session_id for s in sessions)
            print(f"Session found in list: {found}")
            if not found:
                print("Sessions:", sessions)

        # 4. Check session details
        print("Checking session details...")
        async with session.get(f"{base_url}/sessions/{session_id}") as resp:
            details = await resp.json()
            print(f"Session details retrieved. Events count: {len(details.get('events', []))}")
            if len(details.get('events', [])) > 0:
                print("Persistence verified!")
            else:
                print("No events found in persistence yet (might be timing).")

if __name__ == "__main__":
    # Install aiohttp if needed: pip install aiohttp
    try:
        asyncio.run(verify_api())
    except KeyboardInterrupt:
        pass
