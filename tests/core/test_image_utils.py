"""
Tests for image utility functions.
"""
import pytest
import os
import shutil
import tempfile
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
from src.core.image_utils import (
    compress_image,
    compress_image_batch,
    load_image,
    validate_image,
    is_supported_format,
    get_image_info,
    process_image_batch,
    create_llm_optimized_copy,
    ImageValidationError,
    ImageCompressionError
)

@pytest.fixture
def test_image():
    """Create a test image file."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        # Create a small test image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(f.name, 'JPEG')
        return f.name

@pytest.fixture
def test_png_image():
    """Create a temporary PNG test image"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        img.save(tmp.name, 'PNG')
        yield tmp.name
        os.unlink(tmp.name)

def test_is_supported_format():
    """Test supported format checking"""
    assert is_supported_format('test.jpg') is True
    assert is_supported_format('test.jpeg') is True
    assert is_supported_format('test.png') is True
    assert is_supported_format('test.webp') is True
    assert is_supported_format('test.gif') is False
    assert is_supported_format('test.txt') is False

def test_validate_image(test_image):
    """Test image validation"""
    # Test valid image
    is_valid, error = validate_image(test_image)
    assert is_valid is True
    assert error is None
    
    # Test non-existent file
    is_valid, error = validate_image('nonexistent.jpg')
    assert is_valid is False
    assert "not found" in error.lower()
    
    # Create an invalid file for testing
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp.write(b'not an image')
        tmp.flush()
        is_valid, error = validate_image(tmp.name)
        assert is_valid is False
        assert "unsupported format" in error.lower()
        os.unlink(tmp.name)

def test_load_image(test_image):
    """Test image loading"""
    # Test with valid image
    img = load_image(test_image)
    assert isinstance(img, Image.Image)
    assert img.size == (100, 100)
    
    # Test with non-existent file
    with pytest.raises(ImageValidationError):
        load_image('nonexistent.jpg')

def test_get_image_info(test_image):
    """Test getting image information"""
    info = get_image_info(test_image)
    assert isinstance(info, dict)
    assert info['format'] in ['JPEG', 'JPG']
    assert info['mode'] == 'RGB'
    assert info['size'] == (100, 100)
    assert info['width'] == 100
    assert info['height'] == 100
    assert info['file_size'] > 0

def test_compress_image(test_image):
    """Test image compression"""
    output_path = test_image + '_compressed.jpg'
    
    try:
        # Test compression with default values
        result_path = compress_image(test_image, output_path, max_dimension=1920, quality=85)
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0
        
        # Clean up
        os.remove(result_path)
    except Exception as e:
        # Clean up on failure
        if os.path.exists(output_path):
            os.remove(output_path)
        raise e
    finally:
        # Clean up test image
        if os.path.exists(test_image):
            os.remove(test_image)

def test_compress_image_batch(test_image):
    """Test batch image compression"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_files = [test_image]
        
        # Test batch compression with default values
        results = compress_image_batch(
            test_files,
            output_dir=temp_dir,
            quality=85,
            max_dimension=1920
        )
        
        assert len(results) == 1
        assert all(r[0] for r in results)  # Check all operations succeeded
        assert all(os.path.exists(r[1]) for r in results)  # Check all files exist

def test_create_llm_optimized_copy(test_image):
    """Test creating LLM-optimized copy"""
    try:
        temp_dir, compressed_path = create_llm_optimized_copy(
            test_image,
            max_dimension=800,
            quality=85
        )
        
        assert os.path.exists(compressed_path)
        assert os.path.getsize(compressed_path) < os.path.getsize(test_image)
        
        # Verify image dimensions
        with Image.open(compressed_path) as img:
            assert max(img.size) <= 800
    finally:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def test_process_image_batch(test_image):
    """Test batch processing of images"""
    # Create a temporary text file for testing
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp.write(b'not an image')
        tmp.flush()
        
        # Test with mix of valid and invalid files
        test_files = [
            test_image,
            'nonexistent.jpg',
            tmp.name
        ]
        
        results = process_image_batch(test_files)
        assert len(results) == 3
        
        # Check first result (valid image)
        assert results[0][1] is True  # is_valid
        assert results[0][2] is None  # no error
        
        # Check second result (nonexistent file)
        assert results[1][1] is False
        assert "not found" in results[1][2].lower()
        
        # Check third result (invalid format)
        assert results[2][1] is False
        assert "unsupported format" in results[2][2].lower()
        
        os.unlink(tmp.name)

def test_compression_with_invalid_input():
    """Test error handling for invalid input"""
    with pytest.raises(FileNotFoundError):
        compress_image('nonexistent.jpg', 'output.jpg', max_dimension=1920)

@pytest.mark.parametrize("quality,expected_success", [
    (0, True),    # Minimum quality
    (100, True),  # Maximum quality
    (-1, False),  # Invalid quality (too low)
    (101, False), # Invalid quality (too high)
])
def test_compression_quality_bounds(test_image, quality, expected_success):
    """Test compression with different quality values"""
    output_path = test_image + f'_quality_{quality}.jpg'
    
    try:
        if expected_success:
            compress_image(test_image, output_path, quality=quality)
            assert os.path.exists(output_path)
        else:
            with pytest.raises(ValueError):  # PIL raises ValueError for invalid quality
                compress_image(test_image, output_path, quality=quality)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path) 