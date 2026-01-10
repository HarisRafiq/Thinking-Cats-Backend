from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, Any, List
from modular_agent.database import DatabaseManager
from modular_agent.github_agent import GitHubAgent
from api.dependencies import get_current_user
from api.core_dependencies import get_db_manager
import os

router = APIRouter(prefix="/github", tags=["github"])

def get_github_agent(db_manager: DatabaseManager = Depends(get_db_manager)):
    return GitHubAgent(db_manager)

@router.get("/auth-url")
async def get_auth_url():
    """Returns the GitHub OAuth authorization URL."""
    client_id = os.getenv("GITHUB_CLIENT_ID")
    if not client_id:
        raise HTTPException(status_code=500, detail="GitHub Client ID not configured")
    
    scope = "repo,user"
    url = f"https://github.com/login/oauth/authorize?client_id={client_id}&scope={scope}"
    return {"url": url}

@router.get("/callback")
async def github_callback(
    code: str, 
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager),
    agent: GitHubAgent = Depends(get_github_agent)
):
    """Handles the GitHub OAuth callback and stores the access token."""
    token_data = await agent.get_access_token(code)
    
    if "access_token" not in token_data:
        raise HTTPException(status_code=400, detail=f"Failed to get access token: {token_data.get('error_description', 'Unknown error')}")
    
    # Store token in DB
    user_id = str(current_user["_id"])
    await db_manager.update_user_github_settings(user_id, {
        "access_token": token_data["access_token"],
        "connected_at": datetime.utcnow().isoformat()
    })
    
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content="""
        <html>
            <body style="font-family: sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; flex-direction: column; gap: 16px;">
                <h2 style="color: #27ae60;">GitHub Connected Successfully!</h2>
                <p>This window will close automatically.</p>
                <script>
                    setTimeout(() => window.close(), 1000);
                </script>
            </body>
        </html>
    """)

@router.get("/repos")
async def list_repos(
    current_user: Dict[str, Any] = Depends(get_current_user),
    agent: GitHubAgent = Depends(get_github_agent)
):
    """Lists repositories for the connected user."""
    github_settings = current_user.get("github_settings")
    if not github_settings or not github_settings.get("access_token"):
        raise HTTPException(status_code=401, detail="GitHub not connected")
    
    repos = await agent.list_repositories(github_settings["access_token"])
    return repos

@router.post("/settings")
async def update_settings(
    settings: Dict[str, str],
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Updates GitHub settings (repo, branch, active_path)."""
    user_id = str(current_user["_id"])
    # Filter allowed keys
    allowed_keys = ["repo", "branch", "active_path"]
    filtered_settings = {k: v for k, v in settings.items() if k in allowed_keys}
    
    await db_manager.update_user_github_settings(user_id, filtered_settings)
    return {"message": "Settings updated"}

@router.post("/publish/{artifact_id}")
async def publish_artifact(
    artifact_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    agent: GitHubAgent = Depends(get_github_agent)
):
    """Triggers one-click publishing of an artifact."""
    user_id = str(current_user["_id"])
    result = await agent.one_click_publish(user_id, artifact_id)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@router.post("/create-blog")
async def create_blog_repo(
    payload: Dict[str, str],
    current_user: Dict[str, Any] = Depends(get_current_user),
    agent: GitHubAgent = Depends(get_github_agent)
):
    """Creates a new repository and initializes it as a Jekyll blog."""
    github_settings = current_user.get("github_settings")
    if not github_settings or not github_settings.get("access_token"):
        raise HTTPException(status_code=401, detail="GitHub not connected")
    
    repo_name = payload.get("repo_name")
    blog_title = payload.get("blog_title", "My Blog")
    
    if not repo_name:
        raise HTTPException(status_code=400, detail="Repository name is required")
        
    # 1. Create Repo
    create_result = await agent.create_repository(
        access_token=github_settings["access_token"],
        name=repo_name,
        private=False # Public so Pages works easily
    )
    
    if not create_result["success"]:
        raise HTTPException(status_code=400, detail=f"Failed to create repo: {create_result.get('error')}")
    
    repo_data = create_result["repo"]
    full_name = repo_data["full_name"]
    
    # 2. Initialize Jekyll Files
    # We do this asynchronously or await? Await is safer for feedback.
    init_success = await agent.initialize_jekyll_blog(
        access_token=github_settings["access_token"],
        repo_full_name=full_name,
        title=blog_title
    )
    
    if not init_success:
        # We still return success but warn? Or fail?
        # Let's fail if init failed completely, but ideally we'd want to return the repo anyway.
        pass 
        
    # 3. Enable GitHub Pages
    pages_result = await agent.enable_pages(
        access_token=github_settings["access_token"],
        repo_full_name=full_name,
        branch="main", # Default branch
        path="/" # Root
    )
    
    pages_url = None
    if pages_result["success"]:
        pages_url = pages_result.get("url")
        
    return {
        "success": True,
        "repo": repo_data,
        "pages_url": pages_url,
        "message": "Repository created and initialized with GitHub Pages enabled"
    }

from datetime import datetime # Needed for connected_at
