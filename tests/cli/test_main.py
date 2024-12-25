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
        
    def __iter__(self):
        """Make Args iterable for Click"""
        for key, value in self.__dict__.items():
            if value is True:
                yield f"--{key}"
            elif value is False:
                continue
            elif value is None:
                continue
            else:
                yield f"--{key}"
                yield str(value)

@pytest.mark.asyncio
async def test_cli_single_image(test_image: str, api_key: str):
    """Test processing a single image via CLI"""
    
    # Create mock args
    args = list(Args(
        input=test_image,
        provider="gemini",
        model="gemini-1.5-flash",  # Test with specific model
        api_key=api_key,
        compress=True,
        recursive=False
    ))
    
    # Process image
    result = await process_images(args)
    assert result.exit_code == 0

@pytest.mark.asyncio
async def test_cli_model_selection(test_image: str, api_key: str):
    """Test model selection via CLI"""
    
    # Test with default model
    args = list(Args(
        input=test_image,
        provider="gemini",
        api_key=api_key,
        compress=True,
        recursive=False
    ))
    
    result = await process_images(args)
    assert result.exit_code == 0

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 