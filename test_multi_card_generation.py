"""
Test script for multi-card image generation.
Tests template-based image generation with Gemini.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add modular_agent to path
sys.path.append(os.path.join(os.path.dirname(__file__), "modular_agent"))

from modular_agent.image_generator import ImageGenerator
from modular_agent.utils.template_generator import generate_grid_template, get_template_info
from modular_agent.utils.image_splitter import split_grid_image, validate_grid_image


async def test_template_generation():
    """Test template generation."""
    print("\n=== Testing Template Generation ===")
    
    try:
        # Test different card counts
        for num_cards in [4, 6, 9]:
            print(f"\nGenerating template for {num_cards} cards...")
            
            # Get template info
            info = get_template_info(num_cards)
            print(f"Template info: {info}")
            
            # Generate template
            template_bytes = generate_grid_template(num_cards)
            print(f"✓ Template generated: {len(template_bytes)} bytes")
            
            # Save to file for visual inspection
            output_path = f"test_template_{num_cards}cards.png"
            with open(output_path, "wb") as f:
                f.write(template_bytes)
            print(f"✓ Saved to {output_path}")
        
        print("\n✓ Template generation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Template generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_image_splitting():
    """Test image splitting using template dimensions."""
    print("\n=== Testing Image Splitting (Template-Based) ===")
    
    try:
        # Test with exact grid sizes
        test_cases = [4, 9]  # 2x2 and 3x3 grids
        
        for num_cards in test_cases:
            print(f"\nGenerating template with {num_cards} cards...")
            
            # Generate template
            template_bytes = generate_grid_template(num_cards)
            
            # For testing, we'll just split the template itself
            # In real usage, this would be the Gemini-generated image
            card_bytes_list = await split_grid_image(
                template_bytes,
                num_cards=num_cards,
                verbose=True
            )
            
            print(f"✓ Split into {len(card_bytes_list)} cards")
            
            if len(card_bytes_list) != num_cards:
                print(f"✗ Expected {num_cards} cards but got {len(card_bytes_list)}")
                return False
            
            # Save individual cards
            for i, card_bytes in enumerate(card_bytes_list):
                output_path = f"test_card_{num_cards}cards_{i+1}.png"
                with open(output_path, "wb") as f:
                    f.write(card_bytes)
                if i < 2:  # Only print first 2
                    print(f"✓ Saved card {i+1} to {output_path}")
            
            print(f"✓ All {num_cards} cards saved successfully")
        
        print("\n✓ Image splitting tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Image splitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_pipeline_with_gemini():
    """Test the full multi-card generation pipeline with Gemini API."""
    print("\n=== Testing Full Pipeline with Gemini ===")
    print("Generates images using Gemini and saves them locally")
    
    try:
        # Check if API key is set
        from modular_agent.config import GOOGLE_API_KEY
        if not GOOGLE_API_KEY:
            print("⚠ GOOGLE_API_KEY not set, skipping Gemini test")
            return True
        
        # Import GCS utils
        sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
        from gcs_utils import GCSUtils
        
        # Initialize GCS utils
        bucket_name = os.getenv("GCS_BUCKET_NAME", "thinking-cats-images")
        gcs_utils = GCSUtils(bucket_name)

        # Initialize ImageGenerator
        image_gen = ImageGenerator(
            gcs_utils=gcs_utils,
            bucket_name=bucket_name,
            model_name="gemini-2.5-flash-image",
            verbose=True
        )
        
        # Test multi-card generation
        num_cards = 4
        print(f"\nGenerating {num_cards} cards with Gemini...")
        prompt = "You are given an input empty image of a grid. Generate the following images in the grid cells: a cute kitten playing with a ball of yarn, a scenic mountain landscape during sunrise, a delicious plate of sushi, and a futuristic city skyline at night."
        
        card_urls = await image_gen.generate_cards_from_template(
            prompt=prompt,
            num_cards=num_cards,
            session_id="test_gemini_session",
            slide_id="test_gemini_images",
            image_size=1024,
            card_padding=2
        )
        
        if card_urls and len(card_urls) == num_cards:
            print(f"\n✓ Successfully generated {len(card_urls)} cards!")
            for i, url in enumerate(card_urls):
                print(f"  Card {i+1}: {url}")
            
            # Save URLs to file for inspection
            with open("test_card_urls.txt", "w") as f:
                for i, url in enumerate(card_urls):
                    f.write(f"Card {i+1}: {url}\n")
            print("✓ URLs saved to test_card_urls.txt")
            return True
        else:
            expected = num_cards
            got = len(card_urls) if card_urls else 0
            print(f"\n✗ Failed: expected {expected} cards but got {got}")
            return False
        
    except Exception as e:
        print(f"\n✗ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Multi-Card Image Generation Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test 1: Template Generation
    results.append(await test_template_generation())
    
    # Test 2: Image Splitting
    results.append(await test_image_splitting())
    
    # Test 3: Full Pipeline with Gemini (optional, requires API key)
    print("\n" + "=" * 60)
    run_gemini = input("Run Gemini API test? (y/n): ").lower() == 'y'
    if run_gemini:
        results.append(await test_full_pipeline_with_gemini())
    else:
        print("Skipping Gemini test")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
