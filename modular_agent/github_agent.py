import os
import httpx
import json
import base64
from typing import Dict, Any,  List
from datetime import datetime, timezone, timedelta
from .llm import GeminiProvider

# ... (existing code)



# ... (existing code, note: I need to target initialize_jekyll_blog separately or combine carefully if they are far apart)
# They are far apart. I will split this into two replacement chunks or just do two calls if needed, but ReplaceFileContent supports one block.
# Actually I can't modify multiple blocks with replace_file_content.
# I'll use multi_replace_file_content since I need to touch imports AND transform_to_blog_post AND initialize_jekyll_blog.
from .database import DatabaseManager

class GitHubAgent:
    """
    Agent for interacting with GitHub API and transforming artifacts into blog posts.
    """
    def __init__(self, db_manager: DatabaseManager, model_name: str = "gemini-2.5-flash"):
        self.db_manager = db_manager
        self.llm = GeminiProvider(model_name=model_name)
        self.client_id = os.getenv("GITHUB_CLIENT_ID")
        self.client_secret = os.getenv("GITHUB_CLIENT_SECRET")

    async def get_access_token(self, code: str) -> Dict[str, Any]:
        """Exchanges OAuth code for access token."""
        url = "https://github.com/login/oauth/access_token"
        headers = {"Accept": "application/json"}
        params = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, params=params)
            return response.json()

    async def list_repositories(self, access_token: str) -> List[Dict[str, Any]]:
        """Lists repositories for the authenticated user."""
        url = "https://api.github.com/user/repos"
        headers = {
            "Authorization": f"token {access_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        params = {"sort": "updated", "per_page": 100}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            if response.status_code != 200:
                return []
            return response.json()

    async def transform_to_blog_post(self, content: str, format_type: str = "jekyll") -> str:
        """
        Uses AI to transform raw artifact content into a formatted blog post with frontmatter.
        """
        prompt = f"""
        Transform the following content into a professional blog post formatted for {format_type}.
        Include appropriate frontmatter (YAML) at the top.
        
        Format type: {format_type}
        
        Required Frontmatter fields:
        - title: "Your Title Here" (ALWAYS wrap the title in double quotes to avoid YAML errors)
        - date: {(datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S %z')}
        - categories: [blog]
        - layout: post
        
        Content to transform:
        ---
        {content}
        ---
        
        Return ONLY the final markdown content.
        """
        
        response = await self.llm.generate(prompt)
        # Strip outer code fences if any
        if response.startswith("```markdown"):
            response = response[11:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        return response.strip()

    async def publish_file(self, access_token: str, repo_full_name: str, path: str, content: str, message: str, branch: str = "main") -> Dict[str, Any]:
        """
        Publishes (creates or updates) a file to a GitHub repository.
        """
        url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
        headers = {
            "Authorization": f"token {access_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Check if file exists to get SHA (for updates)
        async with httpx.AsyncClient() as client:
            get_resp = await client.get(url, headers=headers, params={"ref": branch})
            sha = None
            if get_resp.status_code == 200:
                sha = get_resp.json().get("sha")
            
            # Prepare payload
            encoded_content = base64.b64encode(content.encode()).decode()
            payload = {
                "message": message,
                "content": encoded_content,
                "branch": branch
            }
            if sha:
                payload["sha"] = sha
                
            put_resp = await client.put(url, headers=headers, json=payload)
            return put_resp.json()

    async def one_click_publish(self, user_id: str, artifact_id: str) -> Dict[str, Any]:
        """
        The core logic for one-click publishing.
        1. Gets user's GitHub settings.
        2. Gets artifact content.
        3. Transforms content to blog post.
        4. Pushes to GitHub.
        """
        settings = await self.db_manager.get_user_github_settings(user_id)
        if not settings or not settings.get("access_token"):
            return {"success": False, "error": "GitHub not connected"}
        
        repo = settings.get("repo")
        branch = settings.get("branch", "main")
        base_path = settings.get("active_path", "_posts")
        
        if not repo:
            return {"success": False, "error": "No repository selected"}
            
        artifact = await self.db_manager.get_artifact(artifact_id)
        if not artifact:
            return {"success": False, "error": "Artifact not found"}
            
        # Transform content
        blog_content = await self.transform_to_blog_post(artifact["content"])
        
        # Determine filename
        # For Jekyll, filename format is YYYY-MM-DD-title.md
        title_slug = "post" # Default
        # Try to extract title from frontmatter
        import re
        title_match = re.search(r'title:\s*(.*)', blog_content)
        if title_match:
            title_slug = title_match.group(1).strip().lower().replace(" ", "-")
            # Remove special chars
            title_slug = re.sub(r'[^a-zA-Z0-9-]', '', title_slug)
            
        date_str = datetime.now().strftime('%Y-%m-%d')
        filename = f"{date_str}-{title_slug}.md"
        full_path = f"{base_path}/{filename}".replace("//", "/")
        
        # Publish
        result = await self.publish_file(
            access_token=settings["access_token"],
            repo_full_name=repo,
            path=full_path,
            content=blog_content,
            message=f"Publish blog post: {filename}",
            branch=branch
        )
        
        if "content" in result:
             # Update artifact status
            await self.db_manager.set_artifact_status(artifact_id, "published")
            return {
                "success": True, 
                "url": result["content"].get("html_url"),
                "path": full_path
            }
        else:
            return {"success": False, "error": result.get("message", "Unknown error")}

    async def create_repository(self, access_token: str, name: str, private: bool = False) -> Dict[str, Any]:
        """
        Creates a new GitHub repository.
        """
        url = "https://api.github.com/user/repos"
        headers = {
            "Authorization": f"token {access_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        payload = {
            "name": name,
            "private": private,
            "auto_init": True,
            "description": "My awesome blog created with ThinkingCats"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code != 201:
                 try:
                     data = response.json()
                     return {"success": False, "error": data.get("message", response.text)}
                 except:
                     return {"success": False, "error": response.text}
            
            return {"success": True, "repo": response.json()}

    async def initialize_jekyll_blog(self, access_token: str, repo_full_name: str, title: str) -> bool:
        """
        Initializes a new repo with basic Jekyll structure.
        """
        # Define basic files
        # Use json.dumps to ensure title is a valid quoted string for YAML
        safe_title = json.dumps(title)
        
        # Calculate baseurl
        repo_name = repo_full_name.split('/')[-1]
        baseurl = ""
        if not repo_name.endswith(".github.io"):
            baseurl = f"/{repo_name}"
            
        files = {
            "_config.yml": f"title: {safe_title}\ntheme: minima\nbaseurl: \"{baseurl}\"\nplugins:\n  - jekyll-feed\n  - jekyll-seo-tag\n",
            "index.md": "---\nlayout: home\n---\n\nWelcome to my new blog! This site was automatically created using ThinkingCats.\n",
            "about.md": "---\nlayout: page\ntitle: About\npermalink: /about/\n---\n\nThis is my blog created with ThinkingCats!\n"
        }
        
        success = True
        for path, content in files.items():
            result = await self.publish_file(
                access_token=access_token,
                repo_full_name=repo_full_name,
                path=path,
                content=content,
                message=f"Initialize {path}"
            )
            if "content" not in result:
                success = False
                print(f"Failed to publish {path}: {result.get('message')}")
        
        return success

    async def enable_pages(self, access_token: str, repo_full_name: str, branch: str = "main", path: str = "/") -> Dict[str, Any]:
        """
        Enables GitHub Pages for a repository.
        """
        url = f"https://api.github.com/repos/{repo_full_name}/pages"
        headers = {
            "Authorization": f"token {access_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        payload = {
            "source": {
                "branch": branch,
                "path": path
            }
        }
        
        async with httpx.AsyncClient() as client:
            # First check if already enabled
            check_resp = await client.get(url, headers=headers)
            if check_resp.status_code == 200:
                return {"success": True, "url": check_resp.json().get("html_url")}
                
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code not in [201, 204]:
                 try:
                     data = response.json()
                     return {"success": False, "error": data.get("message", response.text)}
                 except:
                     return {"success": False, "error": response.text}
            
            return {"success": True, "url": response.json().get("html_url")}
