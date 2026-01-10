import os
from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException, status, Header, Request
from fastapi.security import OAuth2PasswordBearer
import jwt
from api.auth import SECRET_KEY, ALGORITHM, oauth2_scheme
from api.core_dependencies import get_db_manager

# Admin token/key for admin endpoints (set via ADMIN_TOKEN env var)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "admin-token-placeholder")

async def get_current_user(
    request: Request,
    db_manager = Depends(get_db_manager)
) -> Dict[str, Any]:
    """Dependency to get the current authenticated user from header or cookie."""
    # 1. Try to get token from Authorization header
    token = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    
    # 2. Try to get token from cookie if header is missing
    if not token:
        token = request.cookies.get("token")
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        raise credentials_exception
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    user = await db_manager.get_user_by_id(user_id)
    if user is None:
        raise credentials_exception
        
    if user.get("is_blocked"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been blocked. Please contact support.",
        )
        
    return user

async def verify_admin_token(request: Request) -> bool:
    """Dependency to verify admin token for admin endpoints using X-Admin-Token header."""
    # Skip authentication for OPTIONS requests (CORS preflight)
    if request.method == "OPTIONS":
        return True
    
    # Get admin token from custom header
    admin_token = request.headers.get("X-Admin-Token")
    
    if not admin_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-Admin-Token header required",
        )
    
    # Verify token matches admin token
    if admin_token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin token",
        )
    
    return True
