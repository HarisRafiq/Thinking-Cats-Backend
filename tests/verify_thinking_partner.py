import asyncio
import sys
import os
from dotenv import load_dotenv

# Add Backend logic to python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(backend_path)

# Load environment variables
env_path = os.path.join(backend_path, ".env")
if os.path.exists(env_path):
    print(f"Loading environment from {env_path}")
    load_dotenv(env_path)
else:
    print(f"WARNING: .env file not found at {env_path}")

from modular_agent.orchestrator import Orchestrator

async def test_thinking_partner():
    print("Initializing Orchestrator...")
    orchestrator = Orchestrator(model_name="gemini-2.5-flash", verbose=True)
    
    # Test 1: Ambiguous Query
    ambiguous_query = "I want to be rich"
    print(f"\n--- Testing Ambiguous Query: '{ambiguous_query}' ---")
    plan = await orchestrator.generate_plan(ambiguous_query)
    
    print(f"Generated Plan: {plan}")
    
    if not plan:
        print("FAIL: No plan generated.")
        return
    
    first_step = plan[0]
    task_type = first_step.get("task_type")
    objective = first_step.get("objective")
    question = first_step.get("question")
    
    print(f"First Step Task Type: {task_type}")
    print(f"First Step Objective: {objective}")
    print(f"First Step Question: {question}")
    
    # Validation logic
    # We expect 'framework' or 'analysis' task type for structuring
    if task_type in ["framework", "analysis"]:
        print("SUCCESS: Orchestrator chose a structuring task type.")
    else:
        print(f"WARNING: Orchestrator chose '{task_type}' instead of framework/analysis.")
        
    # We expect the question to be directed at the expert, not the user
    if "User" in question or "you" in question.lower().split(): # simplistic check
        # "you" might refer to the expert, so this check is weak. 
        # Better check: does it ask for a framework?
        pass

    # Test 2: Specific Query
    specific_query = "Draft a tweet about thinking cats"
    print(f"\n--- Testing Specific Query: '{specific_query}' ---")
    plan_specific = await orchestrator.generate_plan(specific_query)
    
    if not plan_specific:
        print("FAIL: No plan generated for specific query.")
        return

    first_step_spec = plan_specific[0]
    print(f"Specific Plan First Step: {first_step_spec}")
    if first_step_spec.get("task_type") in ["action", "implementation", "brainstorm"]:
        print("SUCCESS: Orchestrator chose an execution task type.")
    else:
         print(f"WARNING: Orchestrator chose '{first_step_spec.get('task_type')}' for specific task.")

if __name__ == "__main__":
    asyncio.run(test_thinking_partner())
