"""
Artifact Router (Simplified)
Clean API for artifact management with smart generation.
"""
import json
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from api.core_dependencies import get_db_manager
from api.dependencies import get_current_user
from modular_agent.artifact_agent import ArtifactAgent

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


# =====================
# Request Schemas
# =====================

class SuggestionsRequest(BaseModel):
    """Request to get generation suggestions for NEW artifacts only."""
    session_id: str


class GenerateRequest(BaseModel):
    """Request to generate NEW artifact content."""
    session_id: str
    action: Optional[str] = None  # Simple action like "Twitter thread", "Business plan"
    custom_prompt: Optional[str] = None  # User's custom instruction


class MergeRequest(BaseModel):
    """Request to merge session content into existing artifact."""
    session_id: str


class ResolveConflictRequest(BaseModel):
    """Request to resolve a merge conflict."""
    session_id: str
    resolution: str  # User's answer to the conflict question


class UpdateRequest(BaseModel):
    """Request for manual content update."""
    content: Optional[str] = None


class EditRequest(BaseModel):
    """Request for AI-powered artifact editing."""
    instruction: str  # Natural language edit instruction
    model: str = "gemini-3-flash-preview"  # Default model for editing
    auto_commit: bool = False  # If true, commit immediately without preview


class CommitEditRequest(BaseModel):
    """Request to commit a previewed edit."""
    content: str
    plan_summary: Optional[str] = None
    edit_instruction: Optional[str] = None


# =====================
# Endpoints
# =====================

@router.get("")
async def list_artifacts(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """List all artifacts for current user with version counts."""
    artifacts = await db_manager.get_user_artifacts(user_id=str(current_user["_id"]))
    return {"artifacts": artifacts}


@router.get("/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Get artifact details."""
    artifact = await db_manager.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    return artifact


@router.get("/{artifact_id}/versions")
async def get_artifact_versions(
    artifact_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Get artifact version history with session info."""
    artifact = await db_manager.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    versions = await db_manager.get_artifact_versions(artifact_id)
    return {"artifact_id": artifact_id, "versions": versions}


@router.get("/{artifact_id}/versions/{version_id_1}/diff")
async def get_version_diff(
    artifact_id: str,
    version_id_1: str,
    version_id_2: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Get diff between two versions of an artifact."""
    artifact = await db_manager.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    diff_result = await db_manager.get_version_diff(artifact_id, version_id_1, version_id_2)
    if "error" in diff_result:
        raise HTTPException(status_code=404, detail=diff_result["error"])
    
    return diff_result


@router.post("/suggestions")
async def get_suggestions(
    request: SuggestionsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """
    Get AI-generated action suggestions for NEW artifact creation.
    For updating existing artifacts, use the /merge endpoint instead.
    
    Returns simple action strings like ["Twitter thread", "Business plan", "Todo list"].
    """
    user_id = str(current_user["_id"])
    
    # Verify session ownership
    session = await db_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this session")
    
    agent = ArtifactAgent(db_manager=db_manager, model_name=session.get("model"))
    result = await agent.get_suggestions(session_id=request.session_id)
    
    return {
        "actions": result.get("actions", []),
        "allow_custom": result.get("allow_custom", True)
    }


@router.post("/generate")
async def generate_artifact(
    request: GenerateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """
    Generate NEW artifact based on action or custom prompt via SSE.
    For updating existing artifacts, use POST /{artifact_id}/merge instead.
    
    Accepts either:
    - action: Simple string like "Twitter thread", "Business plan"
    - custom_prompt: User's freeform instruction for what to create
    """
    user_id = str(current_user["_id"])
    
    # Verify session ownership
    session = await db_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this session")
    
    # Validate that either action or custom_prompt is provided
    if not request.action and not request.custom_prompt:
        raise HTTPException(status_code=400, detail="Either action or custom_prompt is required")
    
    async def event_generator():
        # Create new artifact
        artifact_id = await db_manager.create_artifact(user_id=user_id)
        yield {
            "event": "artifact_created",
            "data": json.dumps({"artifact_id": artifact_id})
        }
        
        # Set status to generating
        await db_manager.set_artifact_status(artifact_id, "generating")
        
        agent = ArtifactAgent(db_manager=db_manager, model_name=session.get("model"))
        final_content = ""
        checkpoint_count = 0
        checkpoint_interval = 500  # Save checkpoint every 500 characters
        
        try:
            async for event in agent.generate(
                session_id=request.session_id,
                action=request.action,
                custom_prompt=request.custom_prompt
            ):
                event_type = event.get("type", "message")
                
                if event_type == "chunk":
                    chunk_text = event.get("text", "")
                    final_content += chunk_text
                    yield {"event": "chunk", "data": json.dumps({"text": chunk_text})}
                    
                    # Save checkpoint periodically
                    if len(final_content) >= checkpoint_interval * (checkpoint_count + 1):
                        checkpoint_count += 1
                        await db_manager.update_artifact(
                            artifact_id,
                            content=final_content,
                            status="generating"  # Keep as generating during checkpoint
                        )
                elif event_type == "complete":
                    final_content = event.get("content", "")
                elif event_type == "error":
                    # Save partial content before error
                    if final_content:
                        await db_manager.update_artifact(
                            artifact_id,
                            content=final_content,
                            status="draft"
                        )
                    await db_manager.set_artifact_status(artifact_id, "draft")
                    yield {"event": "error", "data": json.dumps(event)}
                    return
                elif event_type == "start":
                    yield {"event": "start", "data": json.dumps(event)}
            
            # Save the generated content
            await db_manager.update_artifact(
                artifact_id,
                content=final_content,
                status="published"
            )
            
            # Create version snapshot linked to this session
            await db_manager.create_artifact_version(
                artifact_id=artifact_id,
                session_id=request.session_id,
                content=final_content
            )
            
            yield {
                "event": "complete",
                "data": json.dumps({
                    "artifact_id": artifact_id,
                    "content": final_content
                })
            }
            
        except Exception as e:
            # Save partial content on exception
            if final_content:
                try:
                    await db_manager.update_artifact(
                        artifact_id,
                        content=final_content,
                        status="draft"
                    )
                except:
                    pass  # Best effort save
            await db_manager.set_artifact_status(artifact_id, "draft")
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
    
    return EventSourceResponse(event_generator())


@router.post("/{artifact_id}/merge")
async def merge_artifact(
    artifact_id: str,
    request: MergeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """
    Auto-merge session content into an existing artifact via SSE.
    
    Automatically integrates new discussion into the artifact.
    If conflicts are detected, returns a clarifying question.
    
    Events:
    - start: Beginning merge analysis
    - chunk: Streaming merged content
    - conflict: Conflict detected, question returned for user
    - complete: Merge successful
    - error: Something went wrong
    """
    user_id = str(current_user["_id"])
    
    # Verify session ownership
    session = await db_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this session")
    
    # Verify artifact ownership
    artifact = await db_manager.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this artifact")
    
    async def event_generator():
        agent = ArtifactAgent(db_manager=db_manager, model_name=session.get("model"))
        final_content = ""
        checkpoint_count = 0
        checkpoint_interval = 500
        
        try:
            async for event in agent.merge_artifact(
                session_id=request.session_id,
                artifact_id=artifact_id
            ):
                event_type = event.get("type", "message")
                
                if event_type == "chunk":
                    chunk_text = event.get("text", "")
                    final_content += chunk_text
                    yield {"event": "chunk", "data": json.dumps({"text": chunk_text})}
                    
                    # Save checkpoint periodically
                    if len(final_content) >= checkpoint_interval * (checkpoint_count + 1):
                        checkpoint_count += 1
                        await db_manager.update_artifact(
                            artifact_id,
                            content=final_content,
                            status="generating"
                        )
                
                elif event_type == "conflict":
                    # Save partial content before conflict
                    if final_content:
                        await db_manager.update_artifact(
                            artifact_id,
                            content=final_content,
                            status="draft"
                        )
                    # Conflict detected - return question and stop
                    yield {"event": "conflict", "data": json.dumps({
                        "question": event.get("question", ""),
                        "suggested_resolutions": event.get("suggested_resolutions", []),
                        "conflict_summary": event.get("conflict_summary", ""),
                        "artifact_id": artifact_id,
                        "session_id": request.session_id
                    })}
                    return
                
                elif event_type == "complete":
                    final_content = event.get("content", "")
                    
                    # Save merged content
                    await db_manager.update_artifact(
                        artifact_id,
                        content=final_content,
                        status="published"
                    )
                    
                    # Create version snapshot
                    await db_manager.create_artifact_version(
                        artifact_id=artifact_id,
                        session_id=request.session_id,
                        content=final_content
                    )
                    
                    yield {"event": "complete", "data": json.dumps({
                        "artifact_id": artifact_id,
                        "content": final_content
                    })}
                
                elif event_type == "error":
                    # Save partial content on error
                    if final_content:
                        await db_manager.update_artifact(
                            artifact_id,
                            content=final_content,
                            status="draft"
                        )
                    yield {"event": "error", "data": json.dumps({"message": event.get("message", "")})}
                    return
                
                elif event_type == "start":
                    yield {"event": "start", "data": json.dumps(event)}
            
        except Exception as e:
            # Save partial content on exception
            if final_content:
                try:
                    await db_manager.update_artifact(
                        artifact_id,
                        content=final_content,
                        status="draft"
                    )
                except:
                    pass
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
    
    return EventSourceResponse(event_generator())


@router.post("/{artifact_id}/resolve")
async def resolve_conflict(
    artifact_id: str,
    request: ResolveConflictRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """
    Resolve a merge conflict with user's clarification via SSE.
    
    Called after /merge returns a conflict event.
    Uses the user's resolution guidance to complete the merge.
    """
    user_id = str(current_user["_id"])
    
    # Verify session ownership
    session = await db_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this session")
    
    # Verify artifact ownership
    artifact = await db_manager.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this artifact")
    
    async def event_generator():
        agent = ArtifactAgent(db_manager=db_manager, model_name=session.get("model"))
        final_content = ""
        
        try:
            async for event in agent.resolve_conflict(
                session_id=request.session_id,
                artifact_id=artifact_id,
                resolution=request.resolution
            ):
                event_type = event.get("type", "message")
                
                if event_type == "chunk":
                    yield {"event": "chunk", "data": json.dumps({"text": event.get("text", "")})}
                
                elif event_type == "complete":
                    final_content = event.get("content", "")
                    
                    # Save resolved content
                    await db_manager.update_artifact(
                        artifact_id,
                        content=final_content,
                        status="published"
                    )
                    
                    # Create version snapshot
                    await db_manager.create_artifact_version(
                        artifact_id=artifact_id,
                        session_id=request.session_id,
                        content=final_content
                    )
                    
                    yield {"event": "complete", "data": json.dumps({
                        "artifact_id": artifact_id,
                        "content": final_content
                    })}
                
                elif event_type == "error":
                    yield {"event": "error", "data": json.dumps({"message": event.get("message", "")})}
                    return
                
                elif event_type == "start":
                    yield {"event": "start", "data": json.dumps(event)}
            
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
    
    return EventSourceResponse(event_generator())


@router.post("/{artifact_id}/edit")
async def edit_artifact(
    artifact_id: str,
    request: EditRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """
    AI-powered artifact editing via SSE.
    
    Streams edit events:
    - thinking: AI is planning the edit
    - edit: Individual surgical edit being applied
    - preview: Full preview of edited content
    - committed: Edit was saved (if auto_commit=true)
    - error: Something went wrong
    """
    user_id = str(current_user["_id"])
    
    # Verify artifact ownership
    artifact = await db_manager.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    async def event_generator():
        import asyncio
        import time
        agent = ArtifactAgent(db_manager=db_manager, model_name=request.model)
        final_content = None
        plan_summary = None
        
        # Create the async generator
        edit_gen = agent.edit_artifact(
            artifact_id=artifact_id,
            instruction=request.instruction
        )
        
        try:
            # Overall timeout for the entire operation (2 minutes)
            overall_timeout = 120.0
            start_time = time.time()
            
            while True:
                # Check overall timeout before getting next event
                current_time = time.time()
                elapsed = current_time - start_time
                if elapsed > overall_timeout:
                    yield {"event": "error", "data": json.dumps({
                        "message": "Edit operation timed out after 2 minutes. Please try again with a simpler request."
                    })}
                    return
                
                # Get next event with timeout (45 seconds per event)
                try:
                    event = await asyncio.wait_for(
                        edit_gen.__anext__(),
                        timeout=45.0
                    )
                except asyncio.TimeoutError:
                    yield {"event": "error", "data": json.dumps({
                        "message": "No response from AI model for 45 seconds. The request may be taking too long."
                    })}
                    return
                except StopAsyncIteration:
                    # Generator exhausted normally
                    break
                
                # Process the event
                event_type = event.get("type", "message")
                
                if event_type == "thinking":
                    yield {"event": "thinking", "data": json.dumps({"message": event.get("message", "")})}
                
                elif event_type == "edit":
                    yield {"event": "edit", "data": json.dumps({
                        "index": event.get("index", 0),
                        "search": event.get("search", ""),
                        "replace": event.get("replace", ""),
                        "is_rewrite": event.get("is_rewrite", False)
                    })}
                
                elif event_type == "edit_skipped":
                    yield {"event": "edit_skipped", "data": json.dumps({
                        "index": event.get("index", 0),
                        "reason": event.get("reason", "")
                    })}
                
                elif event_type == "preview":
                    final_content = event.get("content", "")
                    plan_summary = event.get("plan_summary", "")
                    diff_info = event.get("diff", {})
                    
                    yield {"event": "preview", "data": json.dumps({
                        "content": final_content,
                        "plan_summary": plan_summary,
                        "edit_count": event.get("edit_count", 0),
                        "edit_type": event.get("edit_type", "surgical"),
                        "diff": diff_info
                    })}
                    
                    # Auto-commit if requested
                    if request.auto_commit and final_content:
                        await db_manager.update_artifact(
                            artifact_id,
                            content=final_content,
                            status="published"
                        )
                        
                        # Create edit version
                        version_id = await db_manager.create_edit_version(
                            artifact_id=artifact_id,
                            content=final_content,
                            edit_instruction=request.instruction,
                            plan_summary=plan_summary
                        )
                        
                        yield {"event": "committed", "data": json.dumps({
                            "artifact_id": artifact_id,
                            "version_id": version_id,
                            "plan_summary": plan_summary
                        })}
                
                elif event_type == "error":
                    yield {"event": "error", "data": json.dumps({"message": event.get("message", "")})}
                    return
            
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
    
    return EventSourceResponse(event_generator())


@router.post("/{artifact_id}/commit")
async def commit_edit(
    artifact_id: str,
    request: CommitEditRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """
    Commit a previewed edit to the artifact.
    Called after reviewing the preview from /edit endpoint.
    """
    user_id = str(current_user["_id"])
    
    # Verify artifact ownership
    artifact = await db_manager.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Update artifact
    await db_manager.update_artifact(
        artifact_id,
        content=request.content,
        status="published"
    )
    
    # Create edit version
    version_id = await db_manager.create_edit_version(
        artifact_id=artifact_id,
        content=request.content,
        edit_instruction=request.edit_instruction or "Manual edit",
        plan_summary=request.plan_summary or "Applied edits"
    )
    
    return {
        "artifact_id": artifact_id,
        "version_id": version_id,
        "message": "Edit committed successfully"
    }


@router.put("/{artifact_id}")
async def update_artifact(
    artifact_id: str,
    request: UpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Manual update of artifact content."""
    artifact = await db_manager.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if request.content is not None:
        # Get old content to check if it changed
        old_content = artifact.get("content", "")
        new_content = request.content
        
        # Only create version if content actually changed
        if old_content != new_content:
            await db_manager.update_artifact(
                artifact_id,
                content=new_content,
                status="published"
            )
            
            # Create version snapshot for manual edit
            await db_manager.create_edit_version(
                artifact_id=artifact_id,
                content=new_content,
                edit_instruction="Manual edit",
                plan_summary="User manually edited the artifact"
            )
        else:
            # Content unchanged, just update status if needed
            await db_manager.update_artifact(
                artifact_id,
                content=new_content,
                status="published"
            )
    
    return await db_manager.get_artifact(artifact_id)


@router.delete("/{artifact_id}")
async def delete_artifact(
    artifact_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_manager = Depends(get_db_manager)
):
    """Delete an artifact and all its versions."""
    artifact = await db_manager.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.get("user_id") != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    await db_manager.delete_artifact(artifact_id, str(current_user["_id"]))
    return {"message": "Artifact deleted"}
