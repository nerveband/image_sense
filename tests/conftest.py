import os
import sys
import pytest
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def test_images_dir():
    """Get the test images directory"""
    return project_root / "tests" / "test_images"

@pytest.fixture(scope="session")
def test_metadata_dir():
    """Get the test metadata directory"""
    return project_root / "tests" / "test_metadata"

@pytest.fixture(scope="session")
def bulk_images_dir():
    """Get the bulk test images directory"""
    return project_root / "tests" / "test_images" / "bulk_images_5"

@pytest.fixture(scope="session")
def single_image_dir():
    """Get the single test image directory"""
    return project_root / "tests" / "test_images" / "single"

@pytest.fixture(scope="session")
def temp_output_dir(tmp_path_factory):
    """Create a temporary directory for test outputs"""
    return tmp_path_factory.mktemp("test_outputs")

@pytest.fixture(autouse=True)
def setup_env():
    """Setup environment variables for testing"""
    # Store original env vars
    orig_env = {
        'GOOGLE_API_KEY': os.environ.get('GOOGLE_API_KEY'),
        'ANTHROPIC_API_KEY': os.environ.get('ANTHROPIC_API_KEY')
    }
    
    # Set test env vars if not set
    if not os.environ.get('GOOGLE_API_KEY'):
        os.environ['GOOGLE_API_KEY'] = 'test_google_key'
    if not os.environ.get('ANTHROPIC_API_KEY'):
        os.environ['ANTHROPIC_API_KEY'] = 'test_anthropic_key'
        
    yield
    
    # Restore original env vars
    for key, value in orig_env.items():
        if value is None:
            del os.environ[key]
        else:
            os.environ[key] = value 