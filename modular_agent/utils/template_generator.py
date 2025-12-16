"""
Template generator for creating grid-based image templates.
Generates empty grid templates for multi-image generation.
"""
import cv2
import numpy as np
from typing import Tuple
import io


def generate_grid_template(
    num_cards: int,
    image_size: int = 1024,
    card_padding: int = 10,
    background_color: Tuple[int, int, int] = (240, 240, 240),
    card_color: Tuple[int, int, int] = (255, 255, 255),
    border_color: Tuple[int, int, int] = (200, 200, 200),
    border_width: int = 2
) -> bytes:
    """
    Generate a grid template for multi-image generation.
    
    Args:
        num_cards: Number of cards to generate (will create an NxN grid)
        image_size: Size of the output image (width and height in pixels)
        card_padding: Padding between cards in pixels
        background_color: RGB color for the background
        card_color: RGB color for card cells
        border_color: RGB color for card borders
        border_width: Width of card borders in pixels
        
    Returns:
        PNG image as bytes
        
    Raises:
        ValueError: If num_cards is not a perfect square
    """
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_cards)))
    
    # Create blank image with background color
    img = np.full((image_size, image_size, 3), background_color, dtype=np.uint8)
    
    # Calculate card dimensions
    total_padding = card_padding * (grid_size + 1)
    available_space = image_size - total_padding
    card_size = available_space // grid_size
    
    # Draw cards
    card_count = 0
    for row in range(grid_size):
        for col in range(grid_size):
            if card_count >= num_cards:
                break
            
            # Calculate card position
            x = card_padding + col * (card_size + card_padding)
            y = card_padding + row * (card_size + card_padding)
            
            # Draw card background
            cv2.rectangle(
                img,
                (x, y),
                (x + card_size, y + card_size),
                card_color,
                -1  # Filled
            )
            
            # Draw card border
            cv2.rectangle(
                img,
                (x, y),
                (x + card_size, y + card_size),
                border_color,
                border_width
            )
            
            card_count += 1
        
        if card_count >= num_cards:
            break
    
    # Encode as PNG
    success, encoded_img = cv2.imencode('.png', img)
    if not success:
        raise ValueError("Failed to encode template image")
    
    return encoded_img.tobytes()


def generate_labeled_grid_template(
    num_cards: int,
    labels: list = None,
    image_size: int = 1024,
    card_padding: int = 10,
    background_color: Tuple[int, int, int] = (240, 240, 240),
    card_color: Tuple[int, int, int] = (255, 255, 255),
    border_color: Tuple[int, int, int] = (200, 200, 200),
    border_width: int = 2,
    font_scale: float = 1.0,
    font_color: Tuple[int, int, int] = (100, 100, 100)
) -> bytes:
    """
    Generate a labeled grid template with text placeholders.
    
    Args:
        num_cards: Number of cards to generate
        labels: List of labels for each card (defaults to "Card 1", "Card 2", etc.)
        image_size: Size of the output image
        card_padding: Padding between cards
        background_color: RGB color for background
        card_color: RGB color for cards
        border_color: RGB color for borders
        border_width: Width of borders
        font_scale: Scale of text labels
        font_color: RGB color for text
        
    Returns:
        PNG image as bytes
    """
    # Generate base template
    img_bytes = generate_grid_template(
        num_cards, image_size, card_padding,
        background_color, card_color, border_color, border_width
    )
    
    # Decode image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Generate default labels if not provided
    if labels is None:
        labels = [f"Card {i+1}" for i in range(num_cards)]
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_cards)))
    total_padding = card_padding * (grid_size + 1)
    available_space = image_size - total_padding
    card_size = available_space // grid_size
    
    # Add labels to cards
    font = cv2.FONT_HERSHEY_SIMPLEX
    card_count = 0
    
    for row in range(grid_size):
        for col in range(grid_size):
            if card_count >= num_cards:
                break
            
            # Calculate card position
            x = card_padding + col * (card_size + card_padding)
            y = card_padding + row * (card_size + card_padding)
            
            # Get label text
            label = labels[card_count] if card_count < len(labels) else f"Card {card_count+1}"
            
            # Calculate text position (centered)
            text_size = cv2.getTextSize(label, font, font_scale, 2)[0]
            text_x = x + (card_size - text_size[0]) // 2
            text_y = y + (card_size + text_size[1]) // 2
            
            # Draw text
            cv2.putText(
                img,
                label,
                (text_x, text_y),
                font,
                font_scale,
                font_color,
                2,
                cv2.LINE_AA
            )
            
            card_count += 1
        
        if card_count >= num_cards:
            break
    
    # Encode as PNG
    success, encoded_img = cv2.imencode('.png', img)
    if not success:
        raise ValueError("Failed to encode labeled template image")
    
    return encoded_img.tobytes()


def get_template_info(num_cards: int, image_size: int = 1024, card_padding: int = 10) -> dict:
    """
    Get information about a template without generating it.
    
    Args:
        num_cards: Number of cards
        image_size: Size of the image
        card_padding: Padding between cards
        
    Returns:
        Dictionary with grid_size, card_size, and layout info
    """
    grid_size = int(np.ceil(np.sqrt(num_cards)))
    total_padding = card_padding * (grid_size + 1)
    available_space = image_size - total_padding
    card_size = available_space // grid_size
    
    return {
        "num_cards": num_cards,
        "grid_size": grid_size,
        "card_size": card_size,
        "image_size": image_size,
        "card_padding": card_padding,
        "total_cards_in_grid": grid_size * grid_size
    }
