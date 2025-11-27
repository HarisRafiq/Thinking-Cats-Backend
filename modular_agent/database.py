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
        
        # Initialize subscription and usage fields if new
        if 'subscription_tier' not in user_data:
            user_data['subscription_tier'] = 'free'
            
        # We don't overwrite usage_stats on update, only on insert if missing
        # But since we use $set, we should be careful. 
        # Let's use $setOnInsert for initial fields
        
        # Upsert based on email
        # We split into $set (updates) and $setOnInsert (initials)
        
        update_op = {
            "$set": {
                "email": user_data["email"],
                "name": user_data.get("name"),
                "picture": user_data.get("picture"),
                "google_id": user_data.get("google_id"),
                "updated_at": user_data["updated_at"]
            },
            "$setOnInsert": {
                "created_at": user_data["created_at"],
                "subscription_tier": "free",
                "usage_stats": {
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "message_count": 0,
                    "daily_message_count": 0,
                    "last_reset_date": datetime.utcnow().isoformat()
                }
            }
        }
        
        result = await self.db.users.update_one(
            {"email": user_data["email"]},
            update_op,
            upsert=True
        )
        
        if result.upserted_id:
            return str(result.upserted_id)
            
        # If not upserted, get the existing ID
        user = await self.db.users.find_one({"email": user_data["email"]})
        return str(user["_id"])

    async def update_user_usage(self, user_id: str, input_tokens: int, output_tokens: int, cost: float = 0.0):
        """Updates the user's usage statistics."""
        if self.db is None:
            await self.connect()
            
        # Check if we need to reset daily limits first
        user = await self.get_user_by_id(user_id)
        if user:
            last_reset = user.get("usage_stats", {}).get("last_reset_date")
            if last_reset:
                try:
                    last_reset_date = datetime.fromisoformat(last_reset).date()
                    if datetime.utcnow().date() > last_reset_date:
                        await self.reset_daily_limits(user_id)
                except ValueError:
                    # Handle legacy or invalid date format
                    pass

        await self.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$inc": {
                    "usage_stats.total_tokens": input_tokens + output_tokens,
                    "usage_stats.total_cost": cost,
                    "usage_stats.message_count": 1,
                    "usage_stats.daily_message_count": 1
                },
                "$set": {
                    "updated_at": datetime.utcnow()
                }
            }
        )

    async def check_user_limit(self, user_id: str) -> bool:
        """
        Checks if the user has reached their daily message limit.
        Returns True if user CAN send message, False if limit reached.
        """
        if self.db is None:
            await self.connect()
            
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
            
        tier = user.get("subscription_tier", "free")
        usage = user.get("usage_stats", {})
        
        # Check for daily reset
        last_reset = usage.get("last_reset_date")
        if last_reset:
            try:
                last_reset_date = datetime.fromisoformat(last_reset).date()
                if datetime.utcnow().date() > last_reset_date:
                    # Reset needed - we can do it lazily here or assume it will be done
                    # Let's do it here to be safe and allow the message
                    await self.reset_daily_limits(user_id)
                    return True
            except ValueError:
                pass
        
        daily_count = usage.get("daily_message_count", 0)
        
        # Hardcoded limits for now - move to config later
        LIMITS = {
            "free": 50,
            "pro": 500
        }
        
        limit = LIMITS.get(tier, 50)
        
        return daily_count < limit

    async def reset_daily_limits(self, user_id: str):
        """Resets the daily message count for a user."""
        if self.db is None:
            await self.connect()
            
        await self.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "usage_stats.daily_message_count": 0,
                    "usage_stats.last_reset_date": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow()
                }
            }
        )

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
        
        # Initialize status if not present
        if 'status' not in data:
            data['status'] = 'idle'  # idle, processing, waiting_for_input
        
        # Initialize is_shared if not present
        if 'is_shared' not in data:
            data['is_shared'] = False
            
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

    async def update_session_status(self, session_id: str, status: str, pending_interaction: Optional[Dict[str, Any]] = None):
        """Updates the session status and pending interaction details."""
        if self.db is None:
            await self.connect()
            
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        
        if pending_interaction is not None:
            update_data["pending_interaction"] = pending_interaction
        elif status == "idle" or status == "processing":
            # Clear pending interaction when moving to idle or processing
            update_data["pending_interaction"] = None
            
        await self.db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": update_data}
        )


    async def archive_session(self, session_id: str):
        """Archives a session by setting is_archived to True."""
        if self.db is None:
            await self.connect()
            
        await self.db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"is_archived": True, "updated_at": datetime.utcnow()}}
        )

    async def toggle_session_sharing(self, session_id: str, is_shared: bool) -> bool:
        """Toggles sharing status for a session. Returns True if successful."""
        if self.db is None:
            await self.connect()
        
        result = await self.db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"is_shared": is_shared, "updated_at": datetime.utcnow()}}
        )
        
        return result.modified_count > 0

    async def get_shared_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a session by _id only if it's shared. Returns None if not found or not shared."""
        if self.db is None:
            await self.connect()
        
        try:
            session = await self.db.sessions.find_one({
                "_id": ObjectId(session_id),
                "is_shared": True
            })
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
            print(f"Error retrieving shared session {session_id}: {e}")
            return None

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
