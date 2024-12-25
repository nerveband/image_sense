"""
Test CLI interface for image processing
"""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv
from src.cli.main import process_images

# Load environment variables
load_dotenv()

@pytest.fixture
def test_image() -> str:
    """Get path to a single test image"""
    test_dir = Path("tests/test_images/bulk_images_5")
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend([str(p) for p in test_dir.glob(ext)])
    # Exclude .DS_Store and take first image
    image_paths = [p for p in image_paths if not p.endswith('.DS_Store')]
    return sorted(image_paths)[0]

@pytest.fixture
def api_key() -> str:
    """Get API key from environment"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not found in environment variables")
    return api_key

class Args:
    """Mock CLI arguments"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

@pytest.mark.asyncio
async def test_cli_single_image(test_image: str, api_key: str):
    """Test processing a single image via CLI"""
    
    # Create mock args
    args = Args(
        input=test_image,
        provider="gemini",
        model="gemini-1.5-flash",  # Test with specific model
        api_key=api_key,
        compress=True,
        recursive=False
    )
    
    # Process image
    results = await process_images(args)
    
    # Verify results
    assert len(results) == 1, "Should process exactly one image"
    result = results[0]
    
    assert result['success'], f"Image processing failed: {result.get('error', 'Unknown error')}"
    assert 'metadata' in result, "Result should contain metadata"
    metadata = result['metadata']
    
    # Verify metadata structure
    assert 'description' in metadata, "Metadata should contain description"
    assert 'keywords' in metadata, "Metadata should contain keywords"
    assert 'categories' in metadata, "Metadata should contain categories"
    
    # Verify content
    assert len(metadata['description']) > 0, "Description should not be empty"
    assert len(metadata['keywords']) > 0, "Keywords should not be empty"
    assert len(metadata['categories']) > 0, "Categories should not be empty"

@pytest.mark.asyncio
async def test_cli_model_selection(test_image: str, api_key: str):
    """Test model selection via CLI"""
    
    # Test with default model
    args = Args(
        input=test_image,
        provider="gemini",
        model=None,  # Should use default
        api_key=api_key,
        compress=True,
        recursive=False
    )
    
    results = await process_images(args)
    assert results[0]['success'], "Processing with default model should succeed"
    
    # Test with Gemini 1.5 Flash
    args.model = "gemini-1.5-flash"
    results = await process_images(args)
    assert results[0]['success'], "Processing with Gemini 1.5 Flash should succeed"
    
    # Test with invalid model
    args.model = "invalid-model"
    with pytest.raises(ValueError, match="Unsupported Gemini model"):
        await process_images(args)

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 