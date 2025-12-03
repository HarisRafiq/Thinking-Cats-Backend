from typing import Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from api.auth import SECRET_KEY, ALGORITHM, oauth2_scheme
from api.core_dependencies import get_db_manager

async def get_current_user(token: str = Depends(oauth2_scheme), db_manager = Depends(get_db_manager)) -> Dict[str, Any]:
    """Dependency to get the current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
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
