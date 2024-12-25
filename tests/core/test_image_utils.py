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
    assert isinstance(results[0], dict)
    assert results[0]['path'] == str(test_files[0])
    assert results[0]['success'] is True
    assert 'description' in results[0]
    assert 'keywords' in results[0]
    assert isinstance(results[0]['keywords'], list)
    
    # Second file (nonexistent)
    assert isinstance(results[1], dict)
    assert results[1]['path'] == str(test_files[1])
    assert results[1]['success'] is False
    assert 'error' in results[1]
    assert 'not found' in results[1]['error'].lower()
    
    # Third file (invalid format)
    assert isinstance(results[2], dict)
    assert results[2]['path'] == str(test_files[2])
    assert results[2]['success'] is False
    assert 'error' in results[2]
    assert 'unsupported format' in results[2]['error'].lower()
    
    # Fourth file (valid png)
    assert isinstance(results[3], dict)
    assert results[3]['path'] == str(test_files[3])
    assert results[3]['success'] is True
    assert 'description' in results[3]
    assert 'keywords' in results[3]
    assert isinstance(results[3]['keywords'], list)

def test_process_image_batch_structured_output(setup_test_files):
    """Test batch processing with structured output."""
    # Test with valid files only
    test_files = [
        setup_test_files['valid_images'][0],  # valid jpg
        setup_test_files['valid_images'][1]   # valid png
    ]
    
    results = process_image_batch(test_files)
    
    # Check results structure
    assert len(results) == 2
    
    for result in results:
        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'path' in result
        assert 'description' in result
        assert 'keywords' in result
        assert isinstance(result['keywords'], list)
        
        # Check optional fields if present
        if 'technical_details' in result:
            assert isinstance(result['technical_details'], dict)
            assert 'format' in result['technical_details']
            assert 'dimensions' in result['technical_details']
            
        if 'visual_elements' in result:
            assert isinstance(result['visual_elements'], list)
            
        if 'composition' in result:
            assert isinstance(result['composition'], list)
            
        if 'mood' in result:
            assert isinstance(result['mood'], str)
            
        if 'use_cases' in result:
            assert isinstance(result['use_cases'], list)

def teardown_module():
    """Clean up test data after tests."""
    import shutil
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR) 