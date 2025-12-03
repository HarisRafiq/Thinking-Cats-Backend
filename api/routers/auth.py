from datetime import timedelta
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from api.schemas import AuthRequest, Token
from api.auth import verify_google_token, create_access_token
from api.core_dependencies import get_db_manager
from api.dependencies import get_current_user

router = APIRouter()

@router.post("/auth/google", response_model=Token)
async def login_google(request: AuthRequest, db_manager = Depends(get_db_manager)):
    # Validate request
    if not request.token:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Google authentication token is required",
                "code": "MISSING_TOKEN"
            }
        )
    
    # Verify Google Token
    try:
        google_user = verify_google_token(request.token)
        if not google_user:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid or expired Google authentication token",
                    "code": "INVALID_GOOGLE_TOKEN"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error during Google token verification: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error verifying Google token",
                "code": "TOKEN_VERIFICATION_ERROR"
            }
        )
    
    # Validate required user data
    if not google_user.get("email"):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Google token missing required user information",
                "code": "INCOMPLETE_USER_DATA"
            }
        )
    
    # Create or update user in DB
    try:
        user_data = {
            "email": google_user["email"],
            "name": google_user.get("name"),
            "picture": google_user.get("picture"),
            "google_id": google_user["sub"]
        }
        
        user_id = await db_manager.create_user(user_data)
        if not user_id:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to create user account",
                    "code": "USER_CREATION_FAILED"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Database error while creating user: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Database error while creating user account",
                "code": "DATABASE_ERROR"
            }
        )
    
    # Create JWT
    try:
        access_token_expires = timedelta(minutes=60 * 24 * 7) # 7 days
        access_token = create_access_token(
            data={"sub": user_id}, expires_delta=access_token_expires
        )
        if not access_token:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to create access token",
                    "code": "TOKEN_CREATION_FAILED"
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating access token: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error creating access token",
                "code": "TOKEN_CREATION_ERROR"
            }
        )
    
    # Get full user object
    try:
        user = await db_manager.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "User created but could not be retrieved",
                    "code": "USER_RETRIEVAL_FAILED"
                }
            )
        
        user['id'] = str(user['_id'])
        del user['_id']
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving user: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error retrieving user information",
                "code": "USER_RETRIEVAL_ERROR"
            }
        )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user": user
    }

@router.get("/me")
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current user information with usage stats."""
    user = dict(current_user)
    user['id'] = str(user['_id'])
    del user['_id']
    return user
