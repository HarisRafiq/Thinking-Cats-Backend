import os
import asyncio
from typing import List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime

class DatabaseManager:
    def __init__(self, uri: Optional[str] = None, db_name: str = "thinking_cats"):
        self.uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = db_name
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None

    async def connect(self):
        """Establishes connection to MongoDB."""
        if not self.client:
            try:
                self.client = AsyncIOMotorClient(self.uri)
                self.db = self.client[self.db_name]
                # Ping to verify connection
                await self.client.admin.command('ping')
                print(f"Connected to MongoDB at {self.uri}")
            except Exception as e:
                print(f"Failed to connect to MongoDB: {e}")
                raise e

    async def create_user(self, user_data: Dict[str, Any]) -> str:
        """Creates a new user or updates existing one."""
        if self.db is None:
            await self.connect()
            
        user_data['updated_at'] = datetime.utcnow()
        if 'created_at' not in user_data:
            user_data['created_at'] = datetime.utcnow()
            
        # Upsert based on email
        result = await self.db.users.update_one(
            {"email": user_data["email"]},
            {"$set": user_data},
            upsert=True
        )
        
        if result.upserted_id:
            return str(result.upserted_id)
            
        # If not upserted, get the existing ID
        user = await self.db.users.find_one({"email": user_data["email"]})
        return str(user["_id"])

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieves a user by email."""
        if self.db is None:
            await self.connect()
        return await self.db.users.find_one({"email": email})

    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a user by ID."""
        if self.db is None:
            await self.connect()
        try:
            return await self.db.users.find_one({"_id": ObjectId(user_id)})
        except:
            return None

    async def create_session(self, data: Dict[str, Any], user_id: Optional[str] = None) -> str:
        """Creates a new session and returns the session_id (MongoDB _id as string)."""
        if self.db is None:
            await self.connect()
        
        if user_id:
            data['user_id'] = user_id
            
        data['created_at'] = datetime.utcnow()
        data['updated_at'] = datetime.utcnow()
        
        result = await self.db.sessions.insert_one(data)
        return str(result.inserted_id)

    async def save_session(self, session_id: str, data: Dict[str, Any]):
        """Updates an existing session by _id."""
        if self.db is None:
            await self.connect()
        
        # Ensure updated_at is set
        data['updated_at'] = datetime.utcnow()
        
        await self.db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": data},
            upsert=True
        )

    async def add_message(self, session_id: str, message: Dict[str, Any]):
        """Adds a message to the session's message history."""
        if self.db is None:
            await self.connect()
            
        await self.db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$push": {"messages": message},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )

    async def add_event(self, session_id: str, event: Dict[str, Any]):
        """Adds an event to the session's event stream."""
        if self.db is None:
            await self.connect()
            
        await self.db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$push": {"events": event},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )

    async def add_slide(self, session_id: str, slide: Dict[str, Any]) -> str:
        """Adds a slide to the session's slides array. Generates MongoDB ObjectId for slide id."""
        if self.db is None:
            await self.connect()
        
        # Generate MongoDB ObjectId for slide
        slide_id = ObjectId()
        slide['id'] = str(slide_id)
        
        await self.db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$push": {"slides": slide},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return str(slide_id)

    async def convert_clarification_to_qa(self, session_id: str, selected_option: str) -> Optional[Dict[str, Any]]:
        """
        Converts the most recent unanswered clarification slide to a user_message slide
        with Q&A format. Returns the new slide data or None if no clarification found.
        """
        if self.db is None:
            await self.connect()
        
        # First, find the session and get the clarification slide
        session = await self.db.sessions.find_one(
            {"_id": ObjectId(session_id)},
            {"slides": 1}
        )
        
        if not session or "slides" not in session:
            return None
        
        # Find the last unanswered clarification slide
        clarification_index = None
        clarification_slide = None
        for i, slide in enumerate(session["slides"]):
            if slide.get("type") == "clarification" and not slide.get("answered", False):
                clarification_index = i
                clarification_slide = slide
        
        if clarification_slide is None:
            return None
        
        # Create the new user_message slide with Q&A format
        question = clarification_slide.get("question", "")
        new_slide = {
            "type": "user_message",
            "content": f"Q: {question}\nA: {selected_option}",
            "id": clarification_slide.get("id")  # Keep the same ID
        }
        
        # Replace the clarification slide with the user_message slide
        await self.db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$set": {
                    f"slides.{clarification_index}": new_slide,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return new_slide

    async def archive_session(self, session_id: str):
        """Archives a session by setting is_archived to True."""
        if self.db is None:
            await self.connect()
            
        await self.db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"is_archived": True, "updated_at": datetime.utcnow()}}
        )

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a session by _id."""
        if self.db is None:
            await self.connect()
        
        try:
            session = await self.db.sessions.find_one({"_id": ObjectId(session_id)})
            if session:
                # Convert ObjectId to string for JSON serialization
                session['session_id'] = str(session['_id'])
                del session['_id']
                
                # Convert ObjectIds in slides array to strings
                if 'slides' in session and isinstance(session['slides'], list):
                    for slide in session['slides']:
                        if 'id' in slide and not isinstance(slide['id'], str):
                            slide['id'] = str(slide['id'])
            return session
        except Exception as e:
            print(f"Error retrieving session {session_id}: {e}")
            return None

    async def get_all_sessions(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent sessions."""
        if self.db is None:
            await self.connect()
            
        query = {"is_archived": {"$ne": True}}
        if user_id:
            query['user_id'] = user_id

        cursor = self.db.sessions.find(
            query, 
            {"messages": 0, "events": 0}  # Exclude heavy fields for listing
        ).sort("updated_at", -1).limit(limit)
        
        sessions = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string for each session
        for session in sessions:
            session['session_id'] = str(session['_id'])
            del session['_id']
        
        return sessions

    def close(self):
        """Closes the MongoDB connection."""
        if self.client:
            self.client.close()
