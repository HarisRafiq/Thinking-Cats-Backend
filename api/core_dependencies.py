from modular_agent.database import DatabaseManager
from api.session_manager import SessionManager
import os
import sys

# Import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.gcs_utils import GCSUtils
except ImportError:
    # Fallback if path handling is different
    from modular_agent.utils.gcs_utils import GCSUtils

from modular_agent.image_generator import ImageGenerator
from modular_agent.llm import GeminiProvider

db_manager = DatabaseManager()
session_manager = SessionManager(db_manager)

# Initialize LLM Provider
llm_model_name = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
llm_provider = GeminiProvider(model_name=llm_model_name)

# Initialize Image Generator
bucket_name = os.getenv("GCS_BUCKET_NAME", "thinking-cats-images")
# Using a model that supports image generation
image_model_name = os.getenv("IMAGE_MODEL_NAME", "gemini-2.5-flash-image") 

# Check if we can initialize GCS (might fail if no creds, but we warn)
try:
    gcs_utils = GCSUtils(bucket_name=bucket_name)
    image_generator = ImageGenerator(
        gcs_utils=gcs_utils, 
        bucket_name=bucket_name,
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
        raise Exception("ImageGenerator not initialized")
    return image_generator

