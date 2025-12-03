from modular_agent.database import DatabaseManager
from api.session_manager import SessionManager

db_manager = DatabaseManager()
session_manager = SessionManager(db_manager)

def get_db_manager():
    return db_manager

def get_session_manager():
    return session_manager
