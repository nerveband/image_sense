import os
import pytest
from pathlib import Path

@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables"""
    original_env = dict(os.environ)
    
    # Set up test environment
    test_env = {
        'GOOGLE_API_KEY': os.environ.get('GOOGLE_API_KEY')
    }
    
    # Set test API keys if not present
    if not os.environ.get('GOOGLE_API_KEY'):
        os.environ['GOOGLE_API_KEY'] = 'test_google_key'
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def test_image_path():
    """Get path to test image"""
    return str(Path(__file__).parent / 'test_images' / 'test.jpg')

@pytest.fixture
def test_images_dir():
    """Get path to test images directory"""
    return str(Path(__file__).parent / 'test_images') 