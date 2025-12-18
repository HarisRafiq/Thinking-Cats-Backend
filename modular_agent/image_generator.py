"""
Image generation module using Gemini 3 Pro Image Preview model.
Generates images and uploads them to Google Cloud Storage.
"""
import os
import io
from typing import Optional, Tuple, List
import google.generativeai as genai
from .config import GOOGLE_API_KEY
import sys

# Add utils directory to path for GCSUtils import
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils"))
from gcs_utils import GCSUtils


class ImageGenerator:
    """Generates images using Gemini and uploads them to GCS."""
    
    def __init__(self, gcs_utils: GCSUtils, bucket_name: str, model_name: str, verbose: bool = False):
        """
        Initialize ImageGenerator.
        
        Args:
            gcs_utils: GCSUtils instance for uploading images
            bucket_name: GCS bucket name for constructing public URLs
            model_name: Gemini model name for image generation
            verbose: Enable verbose logging for debugging
        """
        self.gcs_utils = gcs_utils
        self.bucket_name = bucket_name
        self.model_name = model_name
        self.verbose = verbose
        
        # Initialize Gemini model for image generation
        try:
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            print(f"Warning: Could not initialize Gemini image model: {e}")
            self.model = None
    
    async def generate_image(self, prompt: str) -> Optional[Tuple[bytes, str]]:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description for image generation
            
        Returns:
            Tuple of (image_bytes, mime_type) or None if generation fails
        """
        if not self.model:
            print("[ImageGenerator] Model not initialized, skipping image generation")
            return None
            
        try:
            # Generate image using Gemini
            response = await self.model.generate_content_async(prompt)
            
            # Debug: Print response structure
            if self.verbose:
                print(f"[ImageGenerator] Response type: {type(response)}")
                print(f"[ImageGenerator] Response attributes: {dir(response)}")
            
            # Check if response contains image data
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    parts = candidate.content.parts
                    for part in parts:
                        if self.verbose:
                            print(f"[ImageGenerator] Part type: {type(part)}, attributes: {dir(part)}")
                        
                        # Check if part contains image data
                        if hasattr(part, 'inline_data') and part.inline_data:
                            inline_data = part.inline_data
                            if self.verbose:
                                print(f"[ImageGenerator] inline_data attributes: {dir(inline_data)}")
                                if hasattr(inline_data, 'mime_type'):
                                    print(f"[ImageGenerator] inline_data mime_type: {inline_data.mime_type}")
                                if hasattr(inline_data, 'data'):
                                    print(f"[ImageGenerator] inline_data.data type: {type(inline_data.data)}")
                                    print(f"[ImageGenerator] inline_data.data length: {len(inline_data.data) if hasattr(inline_data.data, '__len__') else 'N/A'}")
                            
                            # Return image bytes
                            import base64
                            if hasattr(inline_data, 'data') and inline_data.data:
                                try:
                                    # Check if it's already bytes or base64 string
                                    if isinstance(inline_data.data, bytes):
                                        image_bytes = inline_data.data
                                    else:
                                        # Try to decode as base64
                                        image_bytes = base64.b64decode(inline_data.data)
                                    
                                    if self.verbose:
                                        print(f"[ImageGenerator] Found inline_data, size: {len(image_bytes)} bytes")
                                        print(f"[ImageGenerator] First 100 bytes (hex): {image_bytes[:100].hex()}")
                                    
                                    # Check if it looks like a valid image (PNG starts with 89 50 4E 47)
                                    mime_type = 'image/png'  # default
                                    if len(image_bytes) > 4:
                                        header = image_bytes[:4]
                                        if header == b'\x89PNG':
                                            print(f"[ImageGenerator] ✓ Valid PNG header detected")
                                            mime_type = 'image/png'
                                        elif header[:2] == b'\xff\xd8':
                                            print(f"[ImageGenerator] ✓ Valid JPEG header detected")
                                            mime_type = 'image/jpeg'
                                        else:
                                            print(f"[ImageGenerator] ⚠️  Unknown image format, header: {header.hex()}")
                                    
                                    # Get mime_type from inline_data if available
                                    if hasattr(inline_data, 'mime_type') and inline_data.mime_type:
                                        mime_type = inline_data.mime_type
                                        if self.verbose:
                                            print(f"[ImageGenerator] Using mime_type from response: {mime_type}")
                                    
                                    return (image_bytes, mime_type)
                                except Exception as e:
                                    print(f"[ImageGenerator] Error decoding inline_data: {e}")
                                    if self.verbose:
                                        import traceback
                                        traceback.print_exc()
                        
                        # Alternative: check for image bytes directly
                        if hasattr(part, 'image') and part.image:
                            img_obj = part.image
                            if hasattr(img_obj, 'data'):
                                image_bytes = img_obj.data
                                if self.verbose:
                                    print(f"[ImageGenerator] Found image.data, size: {len(image_bytes)} bytes")
                                # Try to detect format
                                mime_type = 'image/png'
                                if image_bytes[:2] == b'\xff\xd8':
                                    mime_type = 'image/jpeg'
                                return (image_bytes, mime_type)
                            if hasattr(img_obj, 'bytes'):
                                image_bytes = img_obj.bytes
                                if self.verbose:
                                    print(f"[ImageGenerator] Found image.bytes, size: {len(image_bytes)} bytes")
                                # Try to detect format
                                mime_type = 'image/png'
                                if image_bytes[:2] == b'\xff\xd8':
                                    mime_type = 'image/jpeg'
                                return (image_bytes, mime_type)
                            
                            # Check if it's a PIL Image
                            from PIL import Image
                            if isinstance(img_obj, Image.Image):
                                buffer = io.BytesIO()
                                img_obj.save(buffer, format='PNG')
                                image_bytes = buffer.getvalue()
                                if self.verbose:
                                    print(f"[ImageGenerator] Found PIL Image, size: {len(image_bytes)} bytes")
                                return (image_bytes, 'image/png')
            
            # If no image found in response, try alternative approach
            # Some Gemini image models return PIL Images directly
            if hasattr(response, 'image'):
                from PIL import Image
                img = response.image
                if isinstance(img, Image.Image):
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    image_bytes = buffer.getvalue()
                    if self.verbose:
                        print(f"[ImageGenerator] Found response.image (PIL), size: {len(image_bytes)} bytes")
                    return (image_bytes, 'image/png')
            
            # Debug: Print what we got
            print("[ImageGenerator] No image data found in response")
            if hasattr(response, 'text'):
                print(f"[ImageGenerator] Response text: {response.text[:200]}")
            return None
            
        except Exception as e:
            print(f"[ImageGenerator] Error generating image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def generate_image_from_template(self, prompt: str, template_bytes: bytes, template_mime_type: str = "image/png") -> Optional[Tuple[bytes, str]]:
        """
        Generate an image from a text prompt with a template image.
        
        Args:
            prompt: Text description for image generation
            template_bytes: Template image as bytes
            template_mime_type: MIME type of template image
            
        Returns:
            Tuple of (image_bytes, mime_type) or None if generation fails
        """
        if not self.model:
            print("[ImageGenerator] Model not initialized, skipping image generation")
            return None
        
        try:
            # Prepare template image for Gemini
            import base64
            from PIL import Image
            
            # Convert bytes to PIL Image
            template_image = Image.open(io.BytesIO(template_bytes))
            
            if self.verbose:
                print(f"[ImageGenerator] Template image: {template_image.size}, mode: {template_image.mode}")
            
            # Create multimodal prompt with template and text
            # Gemini expects a list of parts: [text, image]
            content_parts = [
                prompt,
                template_image
            ]
            
            if self.verbose:
                print(f"[ImageGenerator] Generating image with template...")
                print(f"[ImageGenerator] Prompt: {prompt[:100]}...")
            
            # Generate image using Gemini with template
            response = await self.model.generate_content_async(content_parts)
            
            # Debug: Print response structure
            if self.verbose:
                print(f"[ImageGenerator] Response type: {type(response)}")
                print(f"[ImageGenerator] Response attributes: {dir(response)}")
            
            # Check if response contains image data (same logic as generate_image)
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    parts = candidate.content.parts
                    for part in parts:
                        if self.verbose:
                            print(f"[ImageGenerator] Part type: {type(part)}, attributes: {dir(part)}")
                        
                        # Check if part contains image data
                        if hasattr(part, 'inline_data') and part.inline_data:
                            inline_data = part.inline_data
                            if self.verbose:
                                print(f"[ImageGenerator] inline_data attributes: {dir(inline_data)}")
                            
                            # Return image bytes
                            if hasattr(inline_data, 'data') and inline_data.data:
                                try:
                                    # Check if it's already bytes or base64 string
                                    if isinstance(inline_data.data, bytes):
                                        image_bytes = inline_data.data
                                    else:
                                        # Try to decode as base64
                                        image_bytes = base64.b64decode(inline_data.data)
                                    
                                    if self.verbose:
                                        print(f"[ImageGenerator] Found inline_data, size: {len(image_bytes)} bytes")
                                    
                                    # Detect mime type
                                    mime_type = 'image/png'
                                    if len(image_bytes) > 4:
                                        header = image_bytes[:4]
                                        if header == b'\x89PNG':
                                            mime_type = 'image/png'
                                        elif header[:2] == b'\xff\xd8':
                                            mime_type = 'image/jpeg'
                                    
                                    # Get mime_type from inline_data if available
                                    if hasattr(inline_data, 'mime_type') and inline_data.mime_type:
                                        mime_type = inline_data.mime_type
                                    
                                    return (image_bytes, mime_type)
                                except Exception as e:
                                    print(f"[ImageGenerator] Error decoding inline_data: {e}")
                                    if self.verbose:
                                        import traceback
                                        traceback.print_exc()
                        
                        # Alternative: check for image bytes directly
                        if hasattr(part, 'image') and part.image:
                            img_obj = part.image
                            if hasattr(img_obj, 'data'):
                                image_bytes = img_obj.data
                                mime_type = 'image/png'
                                if image_bytes[:2] == b'\xff\xd8':
                                    mime_type = 'image/jpeg'
                                return (image_bytes, mime_type)
                            if hasattr(img_obj, 'bytes'):
                                image_bytes = img_obj.bytes
                                mime_type = 'image/png'
                                if image_bytes[:2] == b'\xff\xd8':
                                    mime_type = 'image/jpeg'
                                return (image_bytes, mime_type)
                            
                            # Check if it's a PIL Image
                            if isinstance(img_obj, Image.Image):
                                buffer = io.BytesIO()
                                img_obj.save(buffer, format='PNG')
                                image_bytes = buffer.getvalue()
                                return (image_bytes, 'image/png')
            
            # Try alternative approach for PIL Image response
            if hasattr(response, 'image'):
                img = response.image
                if isinstance(img, Image.Image):
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    image_bytes = buffer.getvalue()
                    return (image_bytes, 'image/png')
            
            print("[ImageGenerator] No image data found in template-based response")
            if hasattr(response, 'text'):
                print(f"[ImageGenerator] Response text: {response.text[:200]}")
            return None
            
        except Exception as e:
            print(f"[ImageGenerator] Error generating image from template: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def generate_and_upload_image(self, prompt: str, session_id: str, slide_id: str) -> Optional[str]:
        """
        Generate an image and upload it to GCS.
        
        Args:
            prompt: Text description for image generation
            session_id: Session ID for organizing images
            slide_id: Slide ID for unique filename
            
        Returns:
            Public GCS URL or None if generation/upload fails
        """
        # Generate image - returns tuple of (image_bytes, mime_type) or None
        result = await self.generate_image(prompt)
        if not result:
            return None
        
        # Handle both old format (just bytes) and new format (tuple)
        if isinstance(result, tuple):
            image_bytes, mime_type = result
        else:
            image_bytes = result
            # Detect mime type from image header
            if image_bytes[:4] == b'\x89PNG':
                mime_type = 'image/png'
            elif image_bytes[:2] == b'\xff\xd8':
                mime_type = 'image/jpeg'
            else:
                mime_type = 'image/png'  # Default
        
        try:
            # Determine file extension from mime type
            if mime_type == 'image/jpeg':
                file_ext = '.jpg'
                content_type = 'image/jpeg'
            elif mime_type == 'image/png':
                file_ext = '.png'
                content_type = 'image/png'
            else:
                file_ext = '.png'
                content_type = mime_type
            
            # Create file path with correct extension
            file_name = f"images/{session_id}/{slide_id}{file_ext}"
            
            # Upload to GCS using BytesIO wrapper with correct content type
            file_data = io.BytesIO(image_bytes)
            # Reset file pointer to beginning
            file_data.seek(0)
            gcs_path = self.gcs_utils.upload_file_to_gcs(file_name, file_data, content_type=content_type)
            
            # Construct public URL
            public_url = f"https://storage.googleapis.com/{self.bucket_name}/{file_name}"
            
            print(f"[ImageGenerator] Successfully uploaded image to {public_url} (type: {content_type})")
            return public_url
            
        except Exception as e:
            print(f"[ImageGenerator] Error uploading image to GCS: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def generate_cards_from_template(
        self,
        prompt: str,
        num_cards: int,
        session_id: str,
        slide_id: str,
        image_size: int = 1024,
        card_padding: int = 10
    ) -> Optional[List[str]]:
        """
        Generate multiple card images from a text prompt using a programmatic template.
        
        Args:
            prompt: Text description for image generation
            num_cards: Number of cards to generate
            session_id: Session ID for organizing images
            slide_id: Slide ID for unique filenames
            image_size: Size of template image (default 1024x1024)
            card_padding: Padding between cards in template (default 10px)
            
        Returns:
            List of public GCS URLs for each card, or None if generation fails
        """
        try:
            # Import utilities
            from .utils.template_generator import generate_grid_template, get_template_info
            from .utils.image_splitter import split_grid_image
            
            # Get template info for logging
            template_info = get_template_info(num_cards, image_size, card_padding)
            print(f"[ImageGenerator] Generating {num_cards} cards in {template_info['grid_size']}x{template_info['grid_size']} grid")
            
            # Step 1: Generate template
            if self.verbose:
                print(f"[ImageGenerator] Generating template for {num_cards} cards...")
            
            template_bytes = generate_grid_template(
                num_cards=num_cards,
                image_size=image_size,
                card_padding=card_padding
            )
            
            if self.verbose:
                print(f"[ImageGenerator] Template generated: {len(template_bytes)} bytes")
            
            # Step 2: Generate composite image using template
            if self.verbose:
                print(f"[ImageGenerator] Generating composite image with prompt...")
            
            result = await self.generate_image_from_template(
                prompt=prompt,
                template_bytes=template_bytes,
                template_mime_type="image/png"
            )
            
            if not result:
                print("[ImageGenerator] Failed to generate composite image")
                return None
            
            composite_bytes, composite_mime_type = result
            
            if self.verbose:
                print(f"[ImageGenerator] Composite image generated: {len(composite_bytes)} bytes")
            
            # Step 3: Split composite image into individual cards
            if self.verbose:
                print(f"[ImageGenerator] Splitting composite image into cards...")
            
            card_bytes_list = await split_grid_image(
                composite_bytes,
                num_cards=num_cards,
                image_size=image_size,
                card_padding=card_padding,
                verbose=self.verbose
            )
            
            print(f"[ImageGenerator] Successfully split into {len(card_bytes_list)} cards")
            
            # Step 4: Upload each card to GCS
            card_urls = []
            for i, card_bytes in enumerate(card_bytes_list):
                try:
                    # Create unique filename for each card
                    file_name = f"images/{session_id}/{slide_id}_card_{i+1}.png"
                    
                    # Upload to GCS
                    file_data = io.BytesIO(card_bytes)
                    file_data.seek(0)
                    gcs_path = self.gcs_utils.upload_file_to_gcs(
                        file_name,
                        file_data,
                        content_type="image/png"
                    )
                    
                    # Construct public URL
                    public_url = f"https://storage.googleapis.com/{self.bucket_name}/{file_name}"
                    card_urls.append(public_url)
                    
                    if self.verbose:
                        print(f"[ImageGenerator] Uploaded card {i+1}/{len(card_bytes_list)}: {public_url}")
                    
                except Exception as e:
                    print(f"[ImageGenerator] Error uploading card {i+1}: {e}")
                    # Continue with other cards even if one fails
                    continue
            
            if not card_urls:
                print("[ImageGenerator] Failed to upload any cards")
                return None
            
            print(f"[ImageGenerator] Successfully uploaded {len(card_urls)} cards")
            return card_urls
            
        except Exception as e:
            print(f"[ImageGenerator] Error in generate_cards_from_template: {e}")
            import traceback
            traceback.print_exc()
            return None

