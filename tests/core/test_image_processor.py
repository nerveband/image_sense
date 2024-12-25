import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from src.core.image_processor import ImageProcessor

@pytest.fixture
def image_processor():
    with patch('src.core.image_processor.get_provider') as mock_provider:
        provider = MagicMock()
        provider.analyze_image = MagicMock(return_value="Test analysis")
        provider.analyze_batch = MagicMock(return_value=[{"description": "Test batch analysis"}])
        mock_provider.return_value = provider
        processor = ImageProcessor(api_key="test_key")
        processor.provider = provider
        yield processor

@pytest.fixture
def mock_compress_image():
    with patch('src.core.image_processor.compress_image') as mock:
        yield mock

def test_image_processor_initialization():
    """Test ImageProcessor initialization with different parameters"""
    with patch('src.core.image_processor.get_provider') as mock_provider:
        processor = ImageProcessor(
            api_key="test_key",
            model="1.5-flash",
            rename_files=True,
            prefix="test_",
            batch_size=5
        )
        assert processor.api_key == "test_key"
        assert processor.model == "1.5-flash"
        assert processor.rename_files is True
        assert processor.prefix == "test_"
        assert processor.batch_size == 5

def test_analyze_image(image_processor, test_image_path):
    """Test analyzing a single image"""
    result = image_processor.analyze_image(test_image_path)
    assert result == "Test analysis"
    image_processor.provider.analyze_image.assert_called_once_with(test_image_path)

def test_analyze_batch(image_processor, test_images_dir):
    """Test analyzing a batch of images"""
    # Create test images
    with tempfile.TemporaryDirectory() as temp_dir:
        test_files = []
        for i in range(2):
            path = os.path.join(temp_dir, f'test{i}.jpg')
            with open(path, 'wb') as f:
                f.write(b'test image data')
            test_files.append(path)
        
        results = image_processor.analyze_batch(test_files)
        assert len(results) == 1
        assert results[0]["description"] == "Test batch analysis"
        image_processor.provider.analyze_batch.assert_called_once()

@pytest.mark.asyncio
async def test_process_images(image_processor, test_images_dir, mock_compress_image):
    """Test processing multiple images"""
    # Create test directory with images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test image
        test_file = os.path.join(temp_dir, 'test.jpg')
        with open(test_file, 'wb') as f:
            f.write(b'test image data')
        
        # Mock the stats to avoid division by zero
        image_processor.stats['processed'] = 1
        image_processor.stats['batch_times'] = [1.0]
        
        # Process images
        results = await image_processor.process_images(temp_dir, compress=True)
        assert isinstance(results, list)

def test_get_image_files(image_processor):
    """Test getting list of image files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        valid_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.gif']:
            path = os.path.join(temp_dir, f'test{ext}')
            with open(path, 'wb') as f:
                f.write(b'test image data')
            if ext != '.gif':  # GIF is not supported
                valid_files.append(path)
        
        # Create an invalid file
        invalid_file = os.path.join(temp_dir, 'test.txt')
        with open(invalid_file, 'w') as f:
            f.write('not an image')
        
        image_files = image_processor._get_image_files(temp_dir)
        assert isinstance(image_files, list)
        assert len(image_files) == len(valid_files)
        assert all(str(f).endswith(('.jpg', '.jpeg', '.png')) for f in image_files)

@pytest.mark.parametrize("model,expected_batch_size", [
    ('1.5-flash', 5),
    ('2-flash', 5),
    ('1.5-pro', 1),
])
def test_batch_size_limits(model, expected_batch_size):
    """Test batch size limits for different models"""
    with patch('src.core.image_processor.get_provider'):
        processor = ImageProcessor(
            api_key="test_key",
            model=model,
            batch_size=10  # Try to set higher than allowed
        )
        assert processor.batch_size <= expected_batch_size

def test_error_handling(image_processor, test_image_path):
    """Test error handling in image processing"""
    error_msg = "Test error"
    image_processor.provider.analyze_image.side_effect = Exception(error_msg)
    
    with pytest.raises(Exception) as exc_info:
        image_processor.analyze_image(test_image_path)
    assert error_msg in str(exc_info.value)

@pytest.mark.asyncio
async def test_progress_callback(image_processor, test_images_dir):
    """Test progress callback functionality"""
    progress_data = []
    def progress_callback(data):
        progress_data.append(data)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test image
        test_file = os.path.join(temp_dir, 'test.jpg')
        with open(test_file, 'wb') as f:
            f.write(b'test image data')
        
        # Mock the stats to avoid division by zero
        image_processor.stats['processed'] = 1
        image_processor.stats['batch_times'] = [1.0]
        
        processor = ImageProcessor(
            api_key="test_key",
            progress_callback=progress_callback
        )
        processor.provider = image_processor.provider
        
        await processor.process_images(temp_dir)
        assert len(progress_data) > 0  # Progress callback should have been called 