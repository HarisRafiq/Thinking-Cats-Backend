import requests
import json
import sseclient

# Start a chat
response = requests.post("http://localhost:8000/chat", json={"problem": "What is the capital of France?"})
if response.status_code != 200:
    print(f"Error starting chat: {response.text}")
    exit(1)

session_id = response.json()["session_id"]
print(f"Session ID: {session_id}")

# Listen to stream
print(f"Connecting to stream: http://localhost:8000/stream/{session_id}")
response = requests.get(f"http://localhost:8000/stream/{session_id}", stream=True)

if response.status_code != 200:
    print(f"Error connecting to stream: {response.status_code}")
    print(response.text)
    exit(1)

client = sseclient.SSEClient(response)
for event in client.events():
    if event.event == "end":
        print("Stream finished")
        break
    elif event.event == "message":
        data = json.loads(event.data)
        print(f"Received event: {data['type']}")
        if data['type'] == 'error':
            print(f"Error Message: {data.get('message')}")
        elif data['type'] == 'consult_end':
            print(f"Response: {data['response'][:50]}...")
    elif event.event == "error":
        print(f"Error: {event.data}")
        break
