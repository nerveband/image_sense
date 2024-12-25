"""
Tests for image utility functions.
"""
import os
import pytest
from PIL import Image

from src.core.image_utils import (
    load_image,
    validate_image,
    ImageValidationError,
    is_supported_format,
    get_image_info,
    process_image_batch
)

# Test data setup
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
os.makedirs(TEST_DATA_DIR, exist_ok=True)

def create_test_image(filename: str, size=(100, 100)) -> str:
    """Create a test image file."""
    path = os.path.join(TEST_DATA_DIR, filename)
    img = Image.new('RGB', size, color='white')
    img.save(path)
    return path

@pytest.fixture(scope="module")
def setup_test_files():
    """Setup test files for all tests."""
    # Create test images
    valid_images = [
        create_test_image('valid1.jpg'),
        create_test_image('valid2.png'),
        create_test_image('valid3.webp')
    ]
    
    # Create invalid file
    invalid_file = os.path.join(TEST_DATA_DIR, 'invalid.txt')
    with open(invalid_file, 'w') as f:
        f.write('not an image')
    
    yield {
        'valid_images': valid_images,
        'invalid_file': invalid_file
    }
    
    # Cleanup happens in teardown_module

def test_is_supported_format():
    """Test supported format checking."""
    assert is_supported_format('test.jpg')
    assert is_supported_format('test.jpeg')
    assert is_supported_format('test.png')
    assert is_supported_format('test.webp')
    assert not is_supported_format('test.gif')
    assert not is_supported_format('test.txt')

def test_validate_image(setup_test_files):
    """Test image validation."""
    # Test with valid image
    is_valid, error = validate_image(setup_test_files['valid_images'][0])
    assert is_valid
    assert error is None

    # Test with non-existent file
    is_valid, error = validate_image('nonexistent.jpg')
    assert not is_valid
    assert 'not found' in error.lower()

    # Test with invalid format
    is_valid, error = validate_image(setup_test_files['invalid_file'])
    assert not is_valid
    assert 'unsupported format' in error.lower()

def test_load_image(setup_test_files):
    """Test image loading."""
    # Test with valid image
    img = load_image(setup_test_files['valid_images'][0])
    assert isinstance(img, Image.Image)
    assert img.size == (100, 100)

    # Test with non-existent file
    with pytest.raises(ImageValidationError):
        load_image('nonexistent.jpg')

    # Test with invalid format
    with pytest.raises(ImageValidationError):
        load_image(setup_test_files['invalid_file'])

def test_get_image_info(setup_test_files):
    """Test getting image information."""
    # Test with valid image
    info = get_image_info(setup_test_files['valid_images'][0])
    
    assert isinstance(info, dict)
    assert info['format'].lower() in ['jpeg', 'jpg']
    assert info['mode'] == 'RGB'
    assert info['size'] == (100, 100)
    assert info['width'] == 100
    assert info['height'] == 100
    assert info['file_size'] > 0

    # Test with non-existent file
    with pytest.raises(ImageValidationError):
        get_image_info('nonexistent.jpg')

def test_process_image_batch(setup_test_files):
    """Test batch processing of images."""
    # Test with mix of valid and invalid files
    test_files = [
        setup_test_files['valid_images'][0],  # valid jpg
        'nonexistent.jpg',                    # missing file
        setup_test_files['invalid_file'],     # invalid format
        setup_test_files['valid_images'][1]   # valid png
    ]
    
    results = process_image_batch(test_files)
    
    # Check results
    assert len(results) == 4
    
    # First file (valid jpg)
    assert results[0][0] == test_files[0]  # file path
    assert results[0][1] is True           # is valid
    assert results[0][2] is None           # no error
    
    # Second file (nonexistent)
    assert results[1][0] == test_files[1]
    assert results[1][1] is False
    assert 'not found' in results[1][2].lower()
    
    # Third file (invalid format)
    assert results[2][0] == test_files[2]
    assert results[2][1] is False
    assert 'unsupported format' in results[2][2].lower()
    
    # Fourth file (valid png)
    assert results[3][0] == test_files[3]
    assert results[3][1] is True
    assert results[3][2] is None

def teardown_module():
    """Clean up test data after tests."""
    import shutil
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR) 