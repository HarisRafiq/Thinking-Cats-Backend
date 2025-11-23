import sys
import os

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Backend.modular_agent.orchestrator import Orchestrator

def test_personality_customization():
    print("Initializing Orchestrator with theme='superhero'...")
    # Mock event callback to capture events
    events = []
    def event_callback(event):
        events.append(event)
        
    orchestrator = Orchestrator(theme="superhero", verbose=True, event_callback=event_callback)
    
    expert_name = "Elon Musk"
    
    print(f"\nConsulting {expert_name}...")
    
    problem = "I want to build a colony on Mars. Ask Elon Musk for advice."
    import asyncio
    response = asyncio.run(orchestrator.process(problem))
    
    print("\n--- Verification Results ---")
    
    # Check events for fictional name
    consult_start_event = next((e for e in events if e["type"] == "consult_start"), None)
    consult_end_event = next((e for e in events if e["type"] == "consult_end"), None)
    
    if consult_start_event:
        print(f"✅ Consult Start Event found.")
        print(f"   Expert Name in Event: {consult_start_event['expert']}")
        
        # Check for sanitization in question
        question_text = consult_start_event['question']
        print(f"   Question Preview: {question_text[:100]}...")
        if "Musk" in question_text and "Elon" in question_text:
             print(f"   ❌ Question contains real name 'Elon Musk'!")
        elif "Musk" in question_text:
             print(f"   ⚠️ Question contains real last name 'Musk'!")
        else:
             print(f"   ✅ Question appears sanitized (no 'Elon Musk').")

        if "Musk" in consult_start_event['expert'] and consult_start_event['expert'] != "Elon Musk":
             print(f"   ✅ Expert name appears to be fictional/themed: {consult_start_event['expert']}")
        else:
             print(f"   ⚠️ Expert name might not be fictional: {consult_start_event['expert']}")
    else:
        print("❌ Consult Start Event NOT found.")

    if consult_end_event:
        print(f"✅ Consult End Event found.")
        print(f"   Expert Name in Event: {consult_end_event['expert']}")
        
        # Check for sanitization in response
        response_text = consult_end_event['response']
        print(f"   Response Preview: {response_text[:100]}...")
        
        if "Musk" in response_text and "Elon" in response_text:
             print(f"   ❌ Response contains real name 'Elon Musk'!")
        elif "Musk" in response_text:
             print(f"   ⚠️ Response contains real last name 'Musk'!")
        else:
             print(f"   ✅ Response appears sanitized (no 'Elon Musk').")
             
    else:
        print("❌ Consult End Event NOT found.")
        
    print("\nTest Complete.")

if __name__ == "__main__":
    test_personality_customization()
