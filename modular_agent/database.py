import os
import asyncio
import difflib
from typing import List, Dict, Any, Optional, Tuple
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
                "is_blocked": False,
                "usage_stats": {
                    "total_tokens": 0,
                    "total_thinking_tokens": 0,
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

    async def update_user_usage(self, user_id: str, input_tokens: int, output_tokens: int, thinking_tokens: int = 0, cost: float = 0.0):
        """Updates the user's usage statistics."""
        if self.db is None:
            await self.connect()
            
        # Check if we need to reset daily limits first
        user = await self.get_user_by_id(user_id)
        if user:
            last_reset = user.get("usage_stats", {}).get("last_reset_date")
            needs_reset = False
            
            if last_reset:
                try:
                    last_reset_date = datetime.fromisoformat(last_reset).date()
                    current_date = datetime.utcnow().date()
                    if current_date > last_reset_date:
                        needs_reset = True
                except (ValueError, TypeError):
                    # Invalid date format - treat as missing and reset
                    needs_reset = True
            else:
                # Missing last_reset_date - initialize it by resetting
                needs_reset = True
            
            if needs_reset:
                await self.reset_daily_limits(user_id)

        await self.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$inc": {
                    "usage_stats.total_tokens": input_tokens + output_tokens + thinking_tokens,
                    "usage_stats.total_thinking_tokens": thinking_tokens,
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
        needs_reset = False
        
        if last_reset:
            try:
                last_reset_date = datetime.fromisoformat(last_reset).date()
                current_date = datetime.utcnow().date()
                if current_date > last_reset_date:
                    needs_reset = True
            except (ValueError, TypeError):
                # Invalid date format - treat as missing and reset
                needs_reset = True
        else:
            # Missing last_reset_date - initialize it by resetting
            needs_reset = True
        
        if needs_reset:
            await self.reset_daily_limits(user_id)
            # After reset, daily_count is 0, so we can allow the message
            return True
        
        daily_count = usage.get("daily_message_count", 0)
        
        # Limits are configurable via environment variables
        FREE_LIMIT = int(os.getenv("DAILY_LIMIT_FREE", "50"))
        PRO_LIMIT = int(os.getenv("DAILY_LIMIT_PRO", "500"))
        LIMITS = {
            "free": FREE_LIMIT,
            "pro": PRO_LIMIT
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
        
        # Initialize usage_stats if not present
        if 'usage_stats' not in data:
            data['usage_stats'] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'thinking_tokens': 0,
                'total_tokens': 0,
                'total_cost': 0.0
            }
            
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

    async def update_session_usage(self, session_id: str, input_tokens: int, output_tokens: int, thinking_tokens: int = 0, cost: float = 0.0):
        """Updates the session's token usage statistics."""
        if self.db is None:
            await self.connect()
        
        await self.db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$inc": {
                    "usage_stats.input_tokens": input_tokens,
                    "usage_stats.output_tokens": output_tokens,
                    "usage_stats.thinking_tokens": thinking_tokens,
                    "usage_stats.total_tokens": input_tokens + output_tokens + thinking_tokens,
                    "usage_stats.total_cost": cost
                },
                "$set": {"updated_at": datetime.utcnow()}
            }
        )

    async def update_session_status(self, session_id: str, status: str, pending_interaction: Optional[Dict[str, Any]] = None):
        """Updates the session status and pending interaction details."""
        if self.db is None:
            await self.connect()
            
        # Get current session to check if we're moving from waiting_for_input
        current_session = await self.get_session(session_id)
        current_status = current_session.get("status") if current_session else None
        is_moving_from_waiting = current_status == "waiting_for_input"
        
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        
        if pending_interaction is not None:
            # Explicitly set pending_interaction
            update_data["pending_interaction"] = pending_interaction
        else:
            # If pending_interaction is not provided:
            # - If moving to idle/processing from waiting_for_input, preserve it
            # - Otherwise, clear it when moving to idle/processing
            if status in ("idle", "processing"):
                if is_moving_from_waiting:
                    # Preserve existing pending_interaction when moving from waiting_for_input
                    # Don't include it in update_data, so it stays as is
                    pass
                else:
                    # Clear pending_interaction when moving to idle/processing from other states
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

    async def get_all_sessions_admin(
        self,
        limit: int = 50,
        skip: int = 0,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        is_shared: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves sessions for admin with pagination and filtering."""
        if self.db is None:
            await self.connect()
            
        query = {"is_archived": {"$ne": True}}
        if user_id:
            query['user_id'] = user_id
        if status:
            query['status'] = status
        if is_shared is not None:
            query['is_shared'] = is_shared

        cursor = self.db.sessions.find(
            query, 
            {"messages": 0, "events": 0}  # Exclude heavy fields for listing
        ).sort("updated_at", -1).skip(skip).limit(limit)
        
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

    def log_llm_call(self, log_data: Dict[str, Any]):
        """Logs LLM calls to local JSONL file with daily rotation for production."""
        import json
        from pathlib import Path
        
        # Create logs directory
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Daily log file with automatic rotation
        log_file = log_dir / f"llm_usage_{datetime.utcnow().date().isoformat()}.jsonl"
        
        try:
            log_data['timestamp'] = datetime.utcnow().isoformat()
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data) + '\n')
        except Exception as e:
            # Don't fail the request if logging fails (fire-and-forget)
            print(f"Warning: Failed to write LLM log: {e}")

    async def get_all_users(
        self, 
        limit: int = 50, 
        skip: int = 0,
        search: Optional[str] = None,
        tier: Optional[str] = None,
        is_blocked: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves all users with pagination and filtering."""
        if self.db is None:
            await self.connect()
        
        query = {}
        
        # Search filter (email or name)
        if search:
            query["$or"] = [
                {"email": {"$regex": search, "$options": "i"}},
                {"name": {"$regex": search, "$options": "i"}}
            ]
        
        # Tier filter
        if tier:
            query["subscription_tier"] = tier
        
        # Blocked status filter
        if is_blocked is not None:
            query["is_blocked"] = is_blocked
            
        cursor = self.db.users.find(query).sort("created_at", -1).skip(skip).limit(limit)
        users = await cursor.to_list(length=limit)
        
        for user in users:
            user['id'] = str(user['_id'])
            del user['_id']
            
        return users

    async def get_llm_logs(
        self, 
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 50, 
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """Retrieves LLM logs with pagination and filtering."""
        if self.db is None:
            await self.connect()
            
        query = {}
        if user_id:
            query['user_id'] = user_id
        if session_id:
            query['session_id'] = session_id
        if model:
            query['model'] = model
            
        cursor = self.db.llm_logs.find(query).sort("timestamp", -1).skip(skip).limit(limit)
        logs = await cursor.to_list(length=limit)
        
        for log in logs:
            log['id'] = str(log['_id'])
            del log['_id']
            # Convert timestamp to string
            if 'timestamp' in log:
                log['timestamp'] = log['timestamp'].isoformat()
            
        return logs

    async def get_admin_stats(self) -> Dict[str, Any]:
        """Retrieves aggregate statistics for the admin dashboard."""
        if self.db is None:
            await self.connect()
            
        total_users = await self.db.users.count_documents({})
        total_sessions = await self.db.sessions.count_documents({})
        
        # Aggregate total tokens and cost from users collection
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_tokens": {"$sum": "$usage_stats.total_tokens"},
                    "total_cost": {"$sum": "$usage_stats.total_cost"}
                }
            }
        ]
        
        usage_stats = await self.db.users.aggregate(pipeline).to_list(length=1)
        
        total_tokens = 0
        total_cost = 0.0
        
        if usage_stats:
            total_tokens = usage_stats[0].get("total_tokens", 0)
            total_cost = usage_stats[0].get("total_cost", 0.0)
            
        return {
            "total_users": total_users,
            "total_sessions": total_sessions,
            "total_tokens": total_tokens,
            "total_cost": total_cost
        }

    async def set_user_status(self, user_id: str, is_blocked: bool) -> bool:
        """Sets the blocked status of a user."""
        if self.db is None:
            await self.connect()
            
        result = await self.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"is_blocked": is_blocked, "updated_at": datetime.utcnow()}}
        )
        return result.modified_count > 0

    async def set_user_tier(self, user_id: str, tier: str) -> bool:
        """Sets the subscription tier of a user."""
        if self.db is None:
            await self.connect()
            
        result = await self.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"subscription_tier": tier, "updated_at": datetime.utcnow()}}
        )
        return result.modified_count > 0

    async def update_user_github_settings(self, user_id: str, settings: Dict[str, Any]) -> bool:
        """Updates user's GitHub settings (access_token, repo, branch, active_path)."""
        if self.db is None:
            await self.connect()
        
        # We store github settings in a nested object
        update_data = {}
        for key, value in settings.items():
            update_data[f"github_settings.{key}"] = value
        
        update_data["updated_at"] = datetime.utcnow()
        
        result = await self.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        return result.modified_count > 0

    async def get_user_github_settings(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves user's GitHub settings."""
        if self.db is None:
            await self.connect()
        
        user = await self.get_user_by_id(user_id)
        if user:
            return user.get("github_settings")
        return None

    async def get_cached_personality(self, expert_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves a cached personality by expert name."""
        if self.db is None:
            await self.connect()
            
        return await self.db.personalities.find_one({"name": expert_name.lower()})

    async def cache_personality(self, expert_name: str, data: Dict[str, Any]):
        """Caches a personality in the database."""
        if self.db is None:
            await self.connect()
            
        data['name'] = expert_name.lower()
        data['updated_at'] = datetime.utcnow()
        
        await self.db.personalities.update_one(
            {"name": expert_name.lower()},
            {"$set": data},
            upsert=True
        )

    # =====================
    # Artifact Methods (Simplified)
    # =====================
    # Schema: artifacts = {_id, user_id, content, status, created_at, updated_at}
    # Schema: artifact_versions = {_id, artifact_id, session_id, content, created_at}
    
    async def create_artifact(self, user_id: str) -> str:
        """Creates a new empty artifact."""
        if self.db is None:
            await self.connect()
        
        artifact = {
            "user_id": user_id,
            "content": "",
            "status": "draft",  # draft, generating, published
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await self.db.artifacts.insert_one(artifact)
        return str(result.inserted_id)

    async def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an artifact by ID with version count."""
        if self.db is None:
            await self.connect()
        
        try:
            artifact = await self.db.artifacts.find_one({"_id": ObjectId(artifact_id)})
            if artifact:
                artifact['artifact_id'] = str(artifact['_id'])
                artifact_id_str = artifact['artifact_id']
                del artifact['_id']
                
                # Get version count
                version_count = await self.db.artifact_versions.count_documents({"artifact_id": artifact_id_str})
                artifact['version_count'] = version_count
            return artifact
        except Exception as e:
            print(f"Error retrieving artifact {artifact_id}: {e}")
            return None

    async def get_user_artifacts(self, user_id: str, limit: int = 50, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves all artifacts for a user with version count."""
        if self.db is None:
            await self.connect()
        
        query = {"user_id": user_id}
        if search:
            query["content"] = {"$regex": search, "$options": "i"}

        # Get artifacts
        cursor = self.db.artifacts.find(query).sort("updated_at", -1).limit(limit)
        artifacts = await cursor.to_list(length=limit)
        
        # Get version counts in one aggregation
        artifact_ids = [a['_id'] for a in artifacts]
        version_counts = {}
        if artifact_ids:
            pipeline = [
                {"$match": {"artifact_id": {"$in": [str(aid) for aid in artifact_ids]}}},
                {"$group": {"_id": "$artifact_id", "count": {"$sum": 1}}}
            ]
            async for doc in self.db.artifact_versions.aggregate(pipeline):
                version_counts[doc['_id']] = doc['count']
        
        for artifact in artifacts:
            artifact['artifact_id'] = str(artifact['_id'])
            artifact['version_count'] = version_counts.get(artifact['artifact_id'], 0)
            del artifact['_id']
        
        return artifacts

    async def update_artifact(self, artifact_id: str, content: str, status: Optional[str] = None) -> bool:
        """Updates artifact content and optionally status."""
        if self.db is None:
            await self.connect()
        
        update = {
            "content": content,
            "updated_at": datetime.utcnow()
        }
        if status:
            update["status"] = status
        
        result = await self.db.artifacts.update_one(
            {"_id": ObjectId(artifact_id)},
            {"$set": update}
        )
        return result.modified_count > 0

    async def set_artifact_status(self, artifact_id: str, status: str) -> bool:
        """Sets artifact status (draft, generating, published)."""
        if self.db is None:
            await self.connect()
        
        result = await self.db.artifacts.update_one(
            {"_id": ObjectId(artifact_id)},
            {"$set": {"status": status, "updated_at": datetime.utcnow()}}
        )
        return result.modified_count > 0

    async def delete_artifact(self, artifact_id: str, user_id: str) -> bool:
        """Deletes artifact and all its versions."""
        if self.db is None:
            await self.connect()
        
        # Delete versions first
        await self.db.artifact_versions.delete_many({"artifact_id": artifact_id})
        
        # Delete artifact
        result = await self.db.artifacts.delete_one({
            "_id": ObjectId(artifact_id),
            "user_id": user_id
        })
        return result.deleted_count > 0

    # =====================
    # Artifact Version Methods
    # =====================
    
    async def create_artifact_version(self, artifact_id: str, session_id: str, content: str) -> str:
        """Creates a version snapshot linked to a session."""
        if self.db is None:
            await self.connect()
        
        version = {
            "artifact_id": artifact_id,
            "session_id": session_id,
            "content": content,
            "version_type": "generation",  # generation = from session
            "created_at": datetime.utcnow()
        }
        
        result = await self.db.artifact_versions.insert_one(version)
        
        # Update artifact timestamp
        await self.db.artifacts.update_one(
            {"_id": ObjectId(artifact_id)},
            {"$set": {"updated_at": datetime.utcnow()}}
        )
        
        return str(result.inserted_id)

    async def create_edit_version(
        self, 
        artifact_id: str, 
        content: str, 
        edit_instruction: str,
        plan_summary: str,
        version_type: str = "edit"  # "edit" = AI edit, "manual" = manual edit
    ) -> str:
        """
        Creates a version snapshot for an edit.
        Edit versions are like 'commits' - each captures a meaningful change.
        """
        if self.db is None:
            await self.connect()
        
        version = {
            "artifact_id": artifact_id,
            "content": content,
            "version_type": version_type,  # "edit" = AI edit, "manual" = manual edit, "generation" = from session
            "edit_instruction": edit_instruction,  # What the user asked for
            "plan_summary": plan_summary,  # What the AI did (commit message)
            "created_at": datetime.utcnow()
        }
        
        result = await self.db.artifact_versions.insert_one(version)
        
        # Update artifact content and timestamp
        await self.db.artifacts.update_one(
            {"_id": ObjectId(artifact_id)},
            {"$set": {
                "content": content,
                "updated_at": datetime.utcnow()
            }}
        )
        
        return str(result.inserted_id)  # Return version ID, not artifact ID

    async def get_session_linked_artifacts(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieves artifacts that have versions linked to this session."""
        if self.db is None:
            await self.connect()
            
        # Find all versions for this session
        cursor = self.db.artifact_versions.find({"session_id": session_id})
        versions = await cursor.to_list(length=100)
        
        artifact_ids = list(set(v["artifact_id"] for v in versions))
        if not artifact_ids:
            return []
            
        # Get the artifacts
        cursor = self.db.artifacts.find({"_id": {"$in": [ObjectId(aid) for aid in artifact_ids]}})
        artifacts = await cursor.to_list(length=len(artifact_ids))
        
        for artifact in artifacts:
            artifact['artifact_id'] = str(artifact['_id'])
            del artifact['_id']
            
        return artifacts

    async def save_artifact_suggestions(self, session_id: str, actions: List[str]):
        """Saves or updates artifact suggestions for a session."""
        if self.db is None:
            await self.connect()
            
        await self.db.artifact_suggestions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "actions": [{"action": a, "selected": False} for a in actions],
                    "updated_at": datetime.utcnow()
                },
                "$setOnInsert": {"created_at": datetime.utcnow()}
            },
            upsert=True
        )

    async def get_artifact_suggestions(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves artifact suggestions for a session."""
        if self.db is None:
            await self.connect()
            
        suggestions = await self.db.artifact_suggestions.find_one({"session_id": session_id})
        if suggestions:
            suggestions['id'] = str(suggestions['_id'])
            del suggestions['_id']
        return suggestions

    async def mark_suggestion_selected(self, session_id: str, action: str):
        """Marks a specific suggestion as selected."""
        if self.db is None:
            await self.connect()
            
        await self.db.artifact_suggestions.update_one(
            {"session_id": session_id, "actions.action": action},
            {"$set": {"actions.$.selected": True, "updated_at": datetime.utcnow()}}
        )

    # =====================
    # Social Generation Methods
    # =====================

    async def create_social_generation(self, session_id: str, prompt: str, cards: List[Dict[str, str]], story: Optional[str] = None, story_type: Optional[str] = None) -> str:
        """
        Creates a new social generation record.
        Schema: {
            session_id: str,
            prompt: str (the master visual prompt or intent),
            story: str (optional narrative summary),
            story_type: str (optional story type used),
            cards: [{url, caption, visual_description}],
            platform_variants: dict (optional platform-optimized versions),
            created_at: datetime
        }
        """
        if self.db is None:
            await self.connect()

        gen_data = {
            "session_id": session_id,
            "prompt": prompt,
            "cards": cards,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        if story:
            gen_data["story"] = story
            
        if story_type:
            gen_data["story_type"] = story_type

        result = await self.db.social_generations.insert_one(gen_data)
        return str(result.inserted_id)

    async def get_social_generations(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieves social generations for a session."""
        if self.db is None:
            await self.connect()

        cursor = self.db.social_generations.find({"session_id": session_id}).sort("created_at", -1).limit(limit)
        gens = await cursor.to_list(length=limit)

        for gen in gens:
            gen['id'] = str(gen['_id'])
            gen['created_at'] = gen['created_at'].isoformat()
            if 'updated_at' in gen:
                gen['updated_at'] = gen['updated_at'].isoformat()
            del gen['_id']

        return gens
    
    async def get_social_generation_by_id(self, generation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single social generation by ID."""
        if self.db is None:
            await self.connect()
        
        from bson import ObjectId
        try:
            gen = await self.db.social_generations.find_one({"_id": ObjectId(generation_id)})
            if gen:
                gen['id'] = str(gen['_id'])
                gen['created_at'] = gen['created_at'].isoformat()
                if 'updated_at' in gen:
                    gen['updated_at'] = gen['updated_at'].isoformat()
                del gen['_id']
            return gen
        except Exception:
            return None
    
    async def update_social_generation_platforms(self, generation_id: str, platform_variants: Dict[str, Any]) -> bool:
        """Updates a social generation with platform-optimized variants."""
        if self.db is None:
            await self.connect()
        
        from bson import ObjectId
        try:
            result = await self.db.social_generations.update_one(
                {"_id": ObjectId(generation_id)},
                {
                    "$set": {
                        "platform_variants": platform_variants,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            return result.modified_count > 0
        except Exception:
            return False

        
        result = await self.db.artifact_versions.insert_one(version)
        
        # Update artifact timestamp
        await self.db.artifacts.update_one(
            {"_id": ObjectId(artifact_id)},
            {"$set": {"updated_at": datetime.utcnow()}}
        )
        
        return str(result.inserted_id)

    async def get_artifact_versions(self, artifact_id: str) -> List[Dict[str, Any]]:
        """Gets all versions for an artifact with session/edit info."""
        if self.db is None:
            await self.connect()
        
        cursor = self.db.artifact_versions.find(
            {"artifact_id": artifact_id}
        ).sort("created_at", -1)
        
        versions = await cursor.to_list(length=100)
        
        # Get session problems for generation versions
        session_ids = list(set(
            v.get("session_id") for v in versions 
            if v.get("session_id") and v.get("version_type", "generation") == "generation"
        ))
        session_map = {}
        if session_ids:
            for sid in session_ids:
                try:
                    session = await self.db.sessions.find_one(
                        {"_id": ObjectId(sid)},
                        {"problem": 1}
                    )
                    if session:
                        session_map[sid] = session.get("problem", "")[:100]
                except:
                    pass
        
        for v in versions:
            v['version_id'] = str(v['_id'])
            v['version_type'] = v.get('version_type', 'generation')  # Default for old versions
            
            if v['version_type'] == 'generation':
                v['session_problem'] = session_map.get(v.get('session_id'), "")
            else:
                # Edit version - use plan_summary as the display text
                v['session_problem'] = v.get('plan_summary', v.get('edit_instruction', 'AI Edit'))
            
            del v['_id']
        
        return versions

    async def get_version_diff(
        self, 
        artifact_id: str, 
        version_id_1: str, 
        version_id_2: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Gets diff between two versions of an artifact.
        If version_id_2 is None, compares against current artifact content.
        
        Returns:
            {
                "diff": unified diff string,
                "added_lines": int,
                "removed_lines": int,
                "changed_lines": int
            }
        """
        if self.db is None:
            await self.connect()
        
        # Get version 1
        version1 = await self.db.artifact_versions.find_one({"_id": ObjectId(version_id_1)})
        if not version1:
            return {"error": "Version 1 not found"}
        
        content1 = version1.get("content", "")
        
        # Get version 2 or current artifact
        if version_id_2:
            version2 = await self.db.artifact_versions.find_one({"_id": ObjectId(version_id_2)})
            if not version2:
                return {"error": "Version 2 not found"}
            content2 = version2.get("content", "")
        else:
            artifact = await self.get_artifact(artifact_id)
            if not artifact:
                return {"error": "Artifact not found"}
            content2 = artifact.get("content", "")
        
        # Generate unified diff
        lines1 = content1.splitlines(keepends=True)
        lines2 = content2.splitlines(keepends=True)
        
        diff = list(difflib.unified_diff(
            lines1,
            lines2,
            fromfile=f"Version {version_id_1[:8]}",
            tofile=f"Version {version_id_2[:8] if version_id_2 else 'Current'}",
            lineterm=''
        ))
        
        # Count changes
        added_lines = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        removed_lines = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
        changed_lines = min(added_lines, removed_lines)  # Approximate
        
        return {
            "diff": ''.join(diff),
            "added_lines": added_lines,
            "removed_lines": removed_lines,
            "changed_lines": changed_lines,
            "diff_lines": diff
        }

    async def get_answered_questions(self, session_id: str, limit: int = 50) -> List[str]:
        """Derives answered questions from agent_response slides (replaces questions_answered array)."""
        if self.db is None:
            await self.connect()
        
        try:
            session = await self.db.sessions.find_one(
                {"_id": ObjectId(session_id)},
                {"slides": 1}
            )
            
            if not session or "slides" not in session:
                return []
            
            questions = []
            for slide in session.get("slides", []):
                if slide.get("type") == "agent_response" and "question" in slide:
                    questions.append(slide["question"])
                    if len(questions) >= limit:
                        break
            
            return questions
        except Exception as e:
            print(f"Error getting answered questions for {session_id}: {e}")
            return []

    async def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Gets session messages and slides for artifact generation context."""
        if self.db is None:
            await self.connect()
        
        try:
            session = await self.db.sessions.find_one(
                {"_id": ObjectId(session_id)},
                {"messages": 1, "slides": 1, "problem": 1}
            )
            if session:
                return {
                    "session_id": str(session['_id']),
                    "problem": session.get("problem", ""),
                    "messages": session.get("messages", [])[-20:],  # Last 20 messages
                    "slides": session.get("slides", [])
                }
            return None
        except Exception as e:
            print(f"Error getting session context {session_id}: {e}")
            return None

    # =====================
    # Deliverables Methods (Block-Based)
    # =====================

    async def create_deliverable(self, user_id: str, deliverable_type: str, title: str, session_id: Optional[str] = None, status: str = "draft") -> str:
        """Create new deliverable with blocks array."""
        if self.db is None:
            await self.connect()
        
        deliverable = {
            "user_id": user_id,
            "session_id": session_id,
            "deliverable_type": deliverable_type,  # "document", "social", "report"
            "title": title,
            "blocks": [],
            "status": status,  # "draft", "generating", "complete", "failed"
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await self.db.deliverables.insert_one(deliverable)
        return str(result.inserted_id)

    async def update_deliverable_status(self, deliverable_id: str, status: str):
        """Update deliverable status."""
        if self.db is None:
            await self.connect()
        
        from bson import ObjectId
        
        await self.db.deliverables.update_one(
            {"_id": ObjectId(deliverable_id)},
            {
                "$set": {
                    "status": status,
                    "updated_at": datetime.utcnow()
                }
            }
        )

    async def add_block(self, deliverable_id: str, block_type: str, content: Dict[str, Any], order: Optional[int] = None) -> str:
        """Add block to deliverable."""
        if self.db is None:
            await self.connect()
        
        from bson import ObjectId
        import uuid
        
        # Get current max order if not specified
        if order is None:
            deliverable = await self.db.deliverables.find_one({"_id": ObjectId(deliverable_id)})
            if deliverable and deliverable.get("blocks"):
                order = max(b.get("order", 0) for b in deliverable["blocks"]) + 1
            else:
                order = 0
        
        block = {
            "block_id": str(uuid.uuid4()),
            "type": block_type,
            "order": order,
            "content": content,
            "metadata": {
                "created_at": datetime.utcnow(),
                "ai_generated": True,
                "copyable": block_type in ["text", "heading", "code", "quote"],
                "downloadable": block_type == "image"
            }
        }
        
        await self.db.deliverables.update_one(
            {"_id": ObjectId(deliverable_id)},
            {
                "$push": {"blocks": block},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return block["block_id"]

    async def update_block(self, deliverable_id: str, block_id: str, content: Dict[str, Any]):
        """Update specific block content."""
        if self.db is None:
            await self.connect()
        
        from bson import ObjectId
        
        await self.db.deliverables.update_one(
            {"_id": ObjectId(deliverable_id), "blocks.block_id": block_id},
            {
                "$set": {
                    "blocks.$.content": content,
                    "blocks.$.metadata.updated_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            }
        )

    async def get_deliverable(self, deliverable_id: str) -> Optional[Dict[str, Any]]:
        """Get deliverable with all blocks."""
        if self.db is None:
            await self.connect()
        
        from bson import ObjectId
        
        deliverable = await self.db.deliverables.find_one({"_id": ObjectId(deliverable_id)})
        if deliverable:
            deliverable["id"] = str(deliverable["_id"])
            del deliverable["_id"]
            # Sort blocks by order
            if "blocks" in deliverable:
                deliverable["blocks"].sort(key=lambda b: b.get("order", 0))
        return deliverable

    async def get_user_deliverables(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all deliverables for a user."""
        if self.db is None:
            await self.connect()
        
        cursor = self.db.deliverables.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        deliverables = await cursor.to_list(length=limit)
        
        for d in deliverables:
            d["id"] = str(d["_id"])
            del d["_id"]
            if "blocks" in d:
                d["blocks"].sort(key=lambda b: b.get("order", 0))
        
        return deliverables

    async def delete_deliverable(self, deliverable_id: str):
        """Delete deliverable and all its blocks."""
        if self.db is None:
            await self.connect()
        
        from bson import ObjectId
        await self.db.deliverables.delete_one({"_id": ObjectId(deliverable_id)})

    async def cache_suggestions(self, session_id: str, suggestions: List[Dict[str, Any]]):
        """Cache deliverable suggestions for a session."""
        if self.db is None:
            await self.connect()
        
        from bson import ObjectId
        
        await self.db.sessions.update_one(
            {"_id": ObjectId(session_id)},
            {
                "$set": {
                    "cached_suggestions": suggestions,
                    "suggestions_cached_at": datetime.utcnow()
                }
            }
        )

    async def get_cached_suggestions(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached suggestions for a session."""
        if self.db is None:
            await self.connect()
        
        from bson import ObjectId
        
        session = await self.db.sessions.find_one(
            {"_id": ObjectId(session_id)},
            {"cached_suggestions": 1, "suggestions_cached_at": 1}
        )
        
        if session and "cached_suggestions" in session:
            return session["cached_suggestions"]
        return None
