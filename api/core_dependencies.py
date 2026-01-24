from fastapi import Depends, HTTPException
from modular_agent.database import DatabaseManager
from modular_agent.block_generator import BlockGenerator
from api.session_manager import SessionManager
import os

from modular_agent.utils.gcs_utils import GCSUtils
from modular_agent.utils.r2_utils import R2Utils
from modular_agent.image_generator import ImageGenerator
from modular_agent.llm import GeminiProvider

db_manager = DatabaseManager()
session_manager = SessionManager(db_manager)

# Initialize LLM Provider
llm_model_name = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
llm_provider = GeminiProvider(model_name=llm_model_name)

# Initialize Image Generator
# R2 Configuration
r2_account_id = os.getenv("R2_ACCOUNT_ID")
r2_access_key_id = os.getenv("R2_ACCESS_KEY_ID")
r2_secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
r2_bucket_name = os.getenv("R2_BUCKET_NAME", "thinking-cats-images")
r2_public_endpoint = os.getenv("R2_PUBLIC_ENDPOINT")

bucket_name = r2_bucket_name # Maintain backward compatibility if needed locally, but primary is now R2

# Using a model that supports image generation
image_model_name = os.getenv("IMAGE_MODEL_NAME", "gemini-2.5-flash-image") 

# Check if we can initialize R2
try:
    if not all([r2_account_id, r2_access_key_id, r2_secret_access_key]):
        raise ValueError("Missing R2 credentials")
        
    r2_utils = R2Utils(
        bucket_name=r2_bucket_name,
        account_id=r2_account_id,
        access_key_id=r2_access_key_id,
        secret_access_key=r2_secret_access_key,
        public_endpoint=r2_public_endpoint
    )
    
    # We still initialize GCSUtils if available for legacy support or other parts, 
    # but ImageGenerator will default to R2 if provided.
    # Note: Originally ImageGenerator took gcs_utils. We will update it to take storage_utils.
    
    image_generator = ImageGenerator(
        storage_utils=r2_utils, 
        bucket_name=r2_bucket_name,
        model_name=image_model_name
    )
except Exception as e:
    print(f"Warning: Failed to initialize ImageGenerator: {e}")
    image_generator = None

def get_db_manager():
    return db_manager

def get_session_manager():
    return session_manager

def get_llm_provider():
    return llm_provider

def get_image_generator():
    if not image_generator:
        raise HTTPException(status_code=503, detail="Image generation service unavailable")
    return image_generator

def get_block_generator(
    db = Depends(get_db_manager),
    llm = Depends(get_llm_provider),
    image_sv = Depends(get_image_generator)
):
    return BlockGenerator(db, llm, image_sv)
