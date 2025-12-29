"""
Image splitting utility for extracting embedded images from grid-based templates.
Uses template dimensions to accurately split composite images.
"""
import cv2
import numpy as np
from typing import List, Optional


async def split_grid_image(
    image_bytes: bytes,
    num_cards: int,
    image_size: int = 1024,
    card_padding: int = 20,
    verbose: bool = False
) -> List[bytes]:
    """
    Split a grid-based image into individual card images.
    Uses template dimensions to guide splitting.
    
    Args:
        image_bytes: Input image as bytes (PNG/JPEG)
        num_cards: Number of cards expected
        image_size: Size of template (1024 default)
        card_padding: Padding used in template
        verbose: Enable verbose logging
        
    Returns:
        List of image bytes for each detected card
        
    Raises:
        ValueError: If splitting fails
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image bytes")
        
        if verbose:
            print(f"[ImageSplitter] Image shape: {img.shape}")
            print(f"[ImageSplitter] Splitting {num_cards} cards with template-based approach")
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_cards)))
        total_padding = card_padding * (grid_size + 1)
        available_space = image_size - total_padding
        card_size = available_space // grid_size
        
        if verbose:
            print(f"[ImageSplitter] Grid: {grid_size}x{grid_size}, Card size: {card_size}x{card_size}")
        
        # Extract cards based on calculated positions
        card_bytes_list = []
        card_count = 0
        
        for row in range(grid_size):
            for col in range(grid_size):
                if card_count >= num_cards:
                    break
                
                # Calculate card position in template
                x = card_padding + col * (card_size + card_padding)
                y = card_padding + row * (card_size + card_padding)
                
                # Scale positions to actual image size if different
                scale_x = img.shape[1] / image_size
                scale_y = img.shape[0] / image_size
                
                x_start = int(x * scale_x)
                y_start = int(y * scale_y)
                x_end = int((x + card_size) * scale_x)
                y_end = int((y + card_size) * scale_y)
                
                # Ensure within bounds
                x_start = max(0, x_start)
                y_start = max(0, y_start)
                x_end = min(img.shape[1], x_end)
                y_end = min(img.shape[0], y_end)
                
                if x_end > x_start and y_end > y_start:
                    card = img[y_start:y_end, x_start:x_end]
                    
                    # Encode as PNG
                    success, encoded_img = cv2.imencode('.png', card)
                    if success:
                        card_bytes_list.append(encoded_img.tobytes())
                        if verbose:
                            print(f"[ImageSplitter] Extracted card {card_count + 1}: [{x_start}:{x_end}, {y_start}:{y_end}]")
                    else:
                        print(f"[ImageSplitter] Warning: Failed to encode card {card_count + 1}")
                
                card_count += 1
            
            if card_count >= num_cards:
                break
        
        if not card_bytes_list:
            raise ValueError("Failed to extract any cards")
        
        print(f"[ImageSplitter] Successfully extracted {len(card_bytes_list)} cards using template dimensions")
        return card_bytes_list
        
    except Exception as e:
        print(f"[ImageSplitter] Error splitting image: {e}")
        import traceback
        traceback.print_exc()
        raise


def validate_grid_image(image_bytes: bytes, expected_count: Optional[int] = None) -> bool:
    """
    Validate that an image can be decoded.
    
    Args:
        image_bytes: Input image as bytes
        expected_count: Expected number of cards (not used in template-based approach)
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img is not None
    except Exception:
        return False
