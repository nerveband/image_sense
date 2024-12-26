import os
import pytest
import asyncio
from pathlib import Path
import numpy as np
from PIL import Image
from unittest.mock import patch
import tempfile

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables"""
    original_env = dict(os.environ)
    
    # Set up test environment
    test_env = {
        'GOOGLE_API_KEY': 'test_google_key',
        'GEMINI_MODEL': 'gemini-2.0-flash-exp'
    }
    
    # Update environment with test values
    os.environ.update(test_env)
    
    # Mock google.generativeai configuration
    with patch('google.generativeai.configure') as mock_configure:
        yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def test_images_dir():
    """Create a temporary directory with test images."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create main directory
        img_dir = Path(tmp_dir) / "test_images"
        img_dir.mkdir()
        
        # Create subdirectory
        sub_dir = img_dir / "subdir"
        sub_dir.mkdir()
        
        # Create test images in both directories
        for dir_path in [img_dir, sub_dir]:
            for i in range(2):
                img_path = dir_path / f"test_{i}.jpg"
                img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
                img.save(img_path)
        
        yield str(img_dir)

@pytest.fixture
def test_image_path():
    """Create a test image file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, "test.jpg")
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(image_path, format='JPEG')
        yield image_path