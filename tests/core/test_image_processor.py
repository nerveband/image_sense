import pytest
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from src.core.image_processor import ImageProcessor
from src.core.image_utils import (
    compress_image,
    compress_image_batch,
    create_llm_optimized_copy,
    create_llm_optimized_batch,
    ImageValidationError,
    ImageCompressionError
)

@pytest.fixture
def test_images(tmp_path):
    """Create test images for testing."""
    image_paths = []
    for i in range(2):
        # Create a random image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save the image
        img_path = tmp_path / f"test{i}.jpg"
        img.save(str(img_path))
        image_paths.append(str(img_path))
    
    return image_paths

@pytest.fixture
def image_processor():
    """Create an ImageProcessor instance for testing."""
    with patch('src.core.image_processor.get_provider') as mock_provider:
        # Create a mock provider
        provider = AsyncMock()
        provider.analyze_image.return_value = {"description": "Test image"}
        provider.analyze_batch.return_value = [{"description": "Test image"}]
        mock_provider.return_value = provider
        
        # Create ImageProcessor with test API key
        processor = ImageProcessor(api_key="test_key")
        processor.provider = provider
        return processor

def test_image_processor_initialization(image_processor):
    """Test ImageProcessor initialization."""
    assert image_processor is not None
    assert image_processor.config is not None
    assert image_processor.api_key == "test_key"

@pytest.mark.asyncio
async def test_analyze_image(test_images, image_processor):
    """Test analyzing a single image."""
    # Test with valid image
    result = await image_processor.analyze_image(test_images[0])
    assert result is not None
    assert isinstance(result, dict)
    assert result["description"] == "Test image"
    
    # Test with invalid image path
    with pytest.raises(Exception):
        await image_processor.analyze_image("nonexistent.jpg")

@pytest.mark.asyncio
async def test_analyze_batch(test_images, image_processor):
    """Test analyzing multiple images."""
    results = await image_processor.analyze_batch(test_images)
    assert len(results) == len(test_images)
    for result in results:
        assert isinstance(result, dict)
        assert result["description"] == "Test image"

@pytest.mark.asyncio
async def test_process_images(test_images, image_processor):
    """Test processing multiple images."""
    # Mock the stats to avoid division by zero
    image_processor.stats['processed'] = 1
    image_processor.stats['batch_times'] = [1.0]
    
    # Mock the process_batch method
    async def mock_process_batch(images, output_dir, compress=None, verbose=False):
        return [{"description": "Test image"} for _ in images]
    
    image_processor._process_image_batch = mock_process_batch
    
    # Test with list of image paths
    results = await image_processor.process_images(test_images)
    assert len(results) == len(test_images)
    
    # Test with directory path
    dir_path = os.path.dirname(test_images[0])
    results = await image_processor.process_images(dir_path)
    assert len(results) == len(test_images)

def test_error_handling(test_images, image_processor):
    """Test error handling."""
    error_msg = "API key not found"
    
    # Test error handling when API key is missing
    with patch('src.core.image_processor.get_provider') as mock_provider:
        mock_provider.side_effect = Exception(error_msg)
        with pytest.raises(Exception) as exc_info:
            image_processor.provider = mock_provider()
            image_processor.analyze_image(test_images[0])
        assert error_msg in str(exc_info.value)

def test_compress_image(test_images, tmp_path):
    output_path = tmp_path / "compressed.jpg"
    
    # Test successful compression
    result = compress_image(
        test_images[0],
        str(output_path),
        max_dimension=800,
        quality=85
    )
    assert os.path.exists(result)
    assert os.path.getsize(result) > 0

    # Test invalid input path
    with pytest.raises(ImageValidationError):
        compress_image(
            "nonexistent.jpg",
            str(output_path),
            max_dimension=800,
            quality=85
        )

    # Test invalid output directory
    invalid_output = "/invalid/path/compressed.jpg"
    with pytest.raises(ImageValidationError):
        compress_image(
            test_images[0],
            invalid_output,
            max_dimension=800,
            quality=85
        )

def test_compress_image_batch(test_images, tmp_path):
    # Test with output directory
    results = compress_image_batch(
        test_images,
        str(tmp_path),
        max_dimension=800,
        quality=85
    )
    
    assert len(results) == len(test_images)
    for original, success, result_path in results:
        assert success
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0

    # Test with invalid paths
    invalid_paths = ["nonexistent1.jpg", "nonexistent2.jpg"]
    results = compress_image_batch(
        invalid_paths,
        str(tmp_path),
        max_dimension=800,
        quality=85
    )
    
    assert len(results) == len(invalid_paths)
    for _, success, error in results:
        assert not success
        assert isinstance(error, str)

def test_create_llm_optimized_copy(test_images):
    temp_dir, compressed_path = create_llm_optimized_copy(
        test_images[0],
        max_dimension=1024,
        quality=85
    )
    
    try:
        assert os.path.exists(compressed_path)
        assert os.path.getsize(compressed_path) > 0
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Test with invalid path
    with pytest.raises(ImageCompressionError):
        create_llm_optimized_copy(
            "nonexistent.jpg",
            max_dimension=1024,
            quality=85
        )

def test_create_llm_optimized_batch(test_images):
    temp_dir, results = create_llm_optimized_batch(
        test_images,
        max_dimension=1024,
        quality=85
    )
    
    try:
        assert len(results) == len(test_images)
        for _, compressed_path in results:
            assert os.path.exists(compressed_path)
            assert os.path.getsize(compressed_path) > 0
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Test with invalid paths
    invalid_paths = ["nonexistent1.jpg", "nonexistent2.jpg"]
    with pytest.raises(ImageCompressionError):
        create_llm_optimized_batch(
            invalid_paths,
            max_dimension=1024,
            quality=85
        )