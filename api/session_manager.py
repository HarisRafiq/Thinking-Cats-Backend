import asyncio
from typing import Dict, Any, Optional
from modular_agent.orchestrator import Orchestrator
from modular_agent.config import DEFAULT_MODEL
from api.events import DoneEvent, ErrorEvent, ProcessingStartEvent, ProcessingCompleteEvent

class SessionManager:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        # Key: session_id, Value: { 'output_queue': asyncio.Queue, 'input_queue': asyncio.Queue, 'task': asyncio.Task }
        self.sessions: Dict[str, Dict[str, Any]] = {}

    async def activate_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """
        Activate a session by creating queues and starting the orchestrator worker.
        Used for resuming sessions that exist in DB but not in memory.
        """
        if session_id in self.sessions:
            # Already active
            return
        
        print(f"Activating session {session_id}...")
        
        # Create queues for the session
        output_queue = asyncio.Queue(maxsize=1000)
        input_queue = asyncio.Queue(maxsize=100)
        
        self.sessions[session_id] = {
            "output_queue": output_queue,
            "input_queue": input_queue,
            "task": None,
            "is_processing": False
        }
        
        # Start the orchestrator worker (without initial problem since we're resuming)
        task = asyncio.create_task(
            self.orchestrator_worker(
                session_id, 
                problem=None,  # No initial problem when resuming
                model=session_data.get("model", DEFAULT_MODEL), 
                input_queue=input_queue, 
                output_queue=output_queue,
                user_id=session_data.get("user_id")
            )
        )
        
        self.sessions[session_id]["task"] = task
        print(f"Session {session_id} activated successfully")

    async def start_new_session(self, session_id: str, problem: str, model: str, user_id: str):
        # Create queues with maxsize to prevent memory issues
        output_queue = asyncio.Queue(maxsize=1000)
        input_queue = asyncio.Queue(maxsize=100)
        
        self.sessions[session_id] = {
            "output_queue": output_queue,
            "input_queue": input_queue,
            "task": None,
            "is_processing": False
        }
        
        # Start the orchestrator in a background task
        task = asyncio.create_task(
            self.orchestrator_worker(
                session_id, 
                problem, 
                model, 
                input_queue, 
                output_queue,
                user_id=user_id
            )
        )
        
        self.sessions[session_id]["task"] = task

    async def orchestrator_worker(self, session_id: str, problem: Optional[str], model: str, input_queue: asyncio.Queue, output_queue: asyncio.Queue, user_id: Optional[str] = None):
        """
        Async worker function that loops to process incoming tasks (solve or chat).
        """
        
        async def callback(event_data: Dict[str, Any]):
            """Put event into async queue with backpressure handling."""
            try:
                # Try to put event immediately (non-blocking if queue has space)
                output_queue.put_nowait(event_data)
            except asyncio.QueueFull:
                # Queue is full - drop oldest event and add new one (backpressure handling)
                try:
                    # Remove oldest event
                    output_queue.get_nowait()
                    # Add new event
                    output_queue.put_nowait(event_data)
                    print(f"[Session {session_id}] Queue full, dropped oldest event")
                except asyncio.QueueEmpty:
                    # Queue was emptied between checks, try again
                    try:
                        output_queue.put_nowait(event_data)
                    except asyncio.QueueFull:
                        # Still full, log and skip this event
                        print(f"[Session {session_id}] Queue persistently full, dropping event: {event_data.get('type')}")

        try:
            orchestrator = Orchestrator(
                model_name=model, 
                verbose=True, 
                event_callback=callback,
                session_id=session_id,
                user_id=user_id,
                db_manager=self.db_manager
            )
            
            # If resuming, we might want to load history here
            await orchestrator.load_session_history()
            
            # Initial process if problem is provided
            if problem:
                # Mark as processing
                if session_id in self.sessions:
                    self.sessions[session_id]["is_processing"] = True
                # Save user message to DB (for agent context)
                await self.db_manager.add_message(session_id, {"role": "user", "content": problem})
                # Emit processing_start event
                processing_start_event = ProcessingStartEvent()
                await callback(processing_start_event.to_dict())
                await orchestrator.process(problem)
                # Emit processing_complete event
                processing_complete_event = ProcessingCompleteEvent()
                await callback(processing_complete_event.to_dict())
                # Mark as not processing
                if session_id in self.sessions:
                    self.sessions[session_id]["is_processing"] = False
            
            # Event loop for follow-up messages
            while True:
                try:
                    # Wait for next input
                    # We use a timeout to check for cancellation or idle
                    task = await asyncio.wait_for(input_queue.get(), timeout=600) # 10 minute timeout
                    
                    if task['type'] == 'message':
                        # Mark as processing
                        if session_id in self.sessions:
                            self.sessions[session_id]["is_processing"] = True
                        # Save user message to DB (for agent context)
                        await self.db_manager.add_message(session_id, {"role": "user", "content": task['content']})
                        
                        # Emit processing_start event
                        processing_start_event = ProcessingStartEvent()
                        await callback(processing_start_event.to_dict())
                        
                        # Note: user_message slide is created inside orchestrator.process()
                        await orchestrator.process(task['content'])
                        
                        # Emit processing_complete event
                        processing_complete_event = ProcessingCompleteEvent()
                        await callback(processing_complete_event.to_dict())
                        # Mark as not processing
                        if session_id in self.sessions:
                            self.sessions[session_id]["is_processing"] = False
                    elif task['type'] == 'stop':
                        break
                except asyncio.TimeoutError:
                    # Idle timeout
                    print(f"Session {session_id} timed out")
                    break
                    
        except asyncio.CancelledError:
            print(f"Session {session_id} worker cancelled")
        except Exception as e:
            # Mark as not processing on error
            if session_id in self.sessions:
                self.sessions[session_id]["is_processing"] = False
            error_event = ErrorEvent(message=str(e))
            await output_queue.put(error_event.to_dict())
        finally:
            # Mark as not processing when worker exits
            if session_id in self.sessions:
                self.sessions[session_id]["is_processing"] = False
            # Signal end of stream
            done_event = DoneEvent()
            await output_queue.put(done_event.to_dict())
            
            # Clean up session
            if session_id in self.sessions:
                del self.sessions[session_id]
                print(f"Session {session_id} cleaned up after worker exit")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)

    def cancel_session(self, session_id: str):
        if session_id in self.sessions:
            task = self.sessions[session_id].get("task")
            if task:
                task.cancel()
            del self.sessions[session_id]
