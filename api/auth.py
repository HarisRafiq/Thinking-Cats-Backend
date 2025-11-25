import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from google.oauth2 import id_token
from google.auth.transport import requests
import jwt
from modular_agent.database import DatabaseManager

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable is required")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_google_token(token: str) -> Optional[Dict[str, Any]]:
    """Verifies a Google ID token and returns the user info."""
    if not token:
        return None
    
    if not GOOGLE_CLIENT_ID:
        print("GOOGLE_CLIENT_ID not configured")
        return None
    
    try:
        # Specify the CLIENT_ID of the app that accesses the backend:
        id_info = id_token.verify_oauth2_token(
            token, 
            requests.Request(), 
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10
        )

        # ID token is valid. Get the user's Google Account ID from the decoded token.
        return id_info
    except ValueError as e:
        # Invalid token
        error_msg = str(e)
        if "Token used too early" in error_msg or "expired" in error_msg.lower():
            print(f"Token expired or used too early: {e}")
        elif "audience" in error_msg.lower() or "aud" in error_msg.lower():
            print(f"Token audience mismatch: {e}")
        else:
            print(f"Token verification failed: {e}")
        return None
    except Exception as e:
        # Unexpected error during token verification
        print(f"Unexpected error during token verification: {e}")
        return None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Creates a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db_manager: DatabaseManager = Depends(DatabaseManager)) -> Dict[str, Any]:
    """Dependency to get the current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "message": "Could not validate credentials",
            "code": "INVALID_CREDENTIALS"
        },
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "message": "Authentication token is required",
                "code": "MISSING_TOKEN"
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "Invalid token: missing user identifier",
                    "code": "INVALID_TOKEN_FORMAT"
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "message": "Token has expired",
                "code": "TOKEN_EXPIRED"
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "message": f"Invalid token: {str(e)}",
                "code": "INVALID_TOKEN"
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "message": f"Token validation error: {str(e)}",
                "code": "TOKEN_VALIDATION_ERROR"
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        user = await db_manager.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "message": "User not found",
                    "code": "USER_NOT_FOUND"
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
    except Exception as e:
        # Database error
        print(f"Database error while fetching user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Database error while validating user",
                "code": "DATABASE_ERROR"
            },
        )
        
    return user
