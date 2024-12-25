import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.core.metadata_handler import MetadataHandler, MetadataError, ExifTool

@pytest.fixture
def metadata_handler():
    with patch('src.core.metadata_handler.ExifTool') as mock_exiftool:
        handler = MetadataHandler()
        handler.exiftool = mock_exiftool.return_value
        yield handler

@pytest.fixture
def mock_exiftool():
    with patch('src.core.metadata_handler.ExifTool') as mock:
        mock_instance = mock.return_value
        mock_instance.read_metadata.return_value = {'Title': 'Test'}
        yield mock_instance

@pytest.fixture
def sample_metadata():
    return {
        'Title': 'Test Image',
        'Description': 'A test image for unit testing',
        'Keywords': ['test', 'unit testing', 'metadata'],
        'Author': 'Test Author'
    }

def test_metadata_handler_initialization():
    """Test MetadataHandler initialization"""
    with patch('src.core.metadata_handler.ExifTool') as mock_exiftool:
        handler = MetadataHandler()
        assert isinstance(handler.exiftool, MagicMock)

def test_read_metadata(metadata_handler, test_image_path):
    """Test reading metadata from an image"""
    metadata_handler.exiftool.read_metadata.return_value = {'Title': 'Test'}
    result = metadata_handler.read_metadata(test_image_path)
    assert result == {'Title': 'Test'}
    metadata_handler.exiftool.read_metadata.assert_called_once_with(test_image_path)

def test_write_metadata(metadata_handler, test_image_path, sample_metadata):
    """Test writing metadata to an image"""
    metadata_handler.write_metadata(test_image_path, sample_metadata)
    metadata_handler.exiftool.write_metadata.assert_called_once_with(
        test_image_path, sample_metadata
    )

def test_copy_metadata(metadata_handler, test_image_path):
    """Test copying metadata between images"""
    source_path = test_image_path
    target_path = str(Path(test_image_path).parent / 'target.jpg')
    tags = ['Title', 'Author']
    
    metadata_handler.copy_metadata(source_path, target_path, tags)
    metadata_handler.exiftool.copy_metadata.assert_called_once_with(
        source_path, target_path, tags
    )

def test_remove_metadata(metadata_handler, test_image_path):
    """Test removing metadata from an image"""
    tags = ['Title', 'Author']
    metadata_handler.remove_metadata(test_image_path, tags)
    metadata_handler.exiftool.remove_metadata.assert_called_once_with(
        test_image_path, tags
    )

def test_process_batch(metadata_handler, test_images_dir, sample_metadata):
    """Test batch processing of images"""
    image_paths = [
        str(Path(test_images_dir) / 'test1.jpg'),
        str(Path(test_images_dir) / 'test2.jpg')
    ]
    
    metadata_handler.exiftool.process_batch.return_value = [
        {'path': p, 'metadata': sample_metadata} for p in image_paths
    ]
    
    results = metadata_handler.process_batch(
        image_paths,
        operation='read',
        metadata=sample_metadata,
        output_format='csv'
    )
    
    assert len(results) == 2
    metadata_handler.exiftool.process_batch.assert_called_once()

def test_metadata_error_handling(metadata_handler, test_image_path):
    """Test error handling in metadata operations"""
    metadata_handler.exiftool.read_metadata.side_effect = MetadataError("Test error")
    
    with pytest.raises(MetadataError) as exc_info:
        metadata_handler.read_metadata(test_image_path)
    assert "Test error" in str(exc_info.value)

@pytest.mark.parametrize("operation,expected_calls", [
    ('read', 'read_metadata'),
    ('write', 'write_metadata'),
    ('remove', 'remove_metadata')
])
def test_batch_operations(metadata_handler, test_images_dir,
                         operation, expected_calls, sample_metadata):
    """Test different batch operations"""
    image_paths = [
        str(Path(test_images_dir) / 'test1.jpg'),
        str(Path(test_images_dir) / 'test2.jpg')
    ]
    
    metadata_handler.process_batch(
        image_paths,
        operation=operation,
        metadata=sample_metadata if operation == 'write' else None
    )
    
    metadata_handler.exiftool.process_batch.assert_called_once() 