"""
Test image compression functionality
"""

import os
import pytest
from PIL import Image
from pathlib import Path
from src.core.llm_handler import LLMHandler

def test_compression():
    """Test image compression"""
    # Get test image path
    test_dir = Path("tests/test_images/bulk_images_5")
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend([str(p) for p in test_dir.glob(ext)])
    # Exclude .DS_Store and take first image
    image_paths = [p for p in image_paths if not p.endswith('.DS_Store')]
    img_path = sorted(image_paths)[0]
    
    # Load original image
    original_size = os.path.getsize(img_path) / 1024  # KB
    print(f"\nOriginal size: {original_size:.1f} KB")
    
    # Create handler instance
    handler = LLMHandler(
        provider="gemini",
        api_key="test-key",  # Not needed for compression test
        compress_images=True
    )
    
    # Compress image
    img = Image.open(img_path)
    compressed_img = handler._compress_image(img)
    
    # Save compressed image to temporary file
    temp_path = "temp_compressed.jpg"
    compressed_img.save(temp_path, "JPEG")
    
    # Get compressed size
    compressed_size = os.path.getsize(temp_path) / 1024  # KB
    print(f"Compressed size: {compressed_size:.1f} KB")
    print(f"Compression ratio: {compressed_size/original_size:.1%}")
    
    # Verify compression
    assert compressed_size < original_size, "Image should be smaller after compression"
    assert compressed_size < 5 * 1024, "Compressed image should be under 5MB"
    
    # Clean up
    os.remove(temp_path)

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 