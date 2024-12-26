import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import exiftool
from src.core.metadata_handler import MetadataHandler, MetadataError

@pytest.fixture
def metadata_handler():
    with patch('exiftool.ExifToolHelper') as mock_exiftool:
        handler = MetadataHandler()
        handler.et = mock_exiftool.return_value
        yield handler

@pytest.fixture
def mock_exiftool():
    with patch('exiftool.ExifToolHelper') as mock:
        mock_instance = mock.return_value
        mock_instance.get_metadata.return_value = [{'Title': 'Test'}]
        yield mock_instance

@pytest.fixture
def sample_metadata():
    return {
        'description': 'A test image for unit testing',
        'keywords': ['test', 'unit testing', 'metadata'],
        'visual_elements': ['element1', 'element2'],
        'composition': ['rule of thirds', 'leading lines'],
        'mood': 'peaceful',
        'use_cases': ['testing', 'documentation'],
        'technical_details': {
            'format': 'JPEG',
            'dimensions': '1920x1080',
            'color_space': 'sRGB'
        }
    }

def test_metadata_handler_initialization():
    """Test MetadataHandler initialization"""
    with patch('exiftool.ExifToolHelper') as mock_exiftool:
        handler = MetadataHandler()
        assert isinstance(handler.et, MagicMock)

def test_read_metadata(metadata_handler, test_image_path):
    """Test reading metadata from an image"""
    metadata_handler.et.get_metadata.return_value = [{'Title': 'Test'}]
    result = metadata_handler.read_metadata(test_image_path)
    assert result['success'] is True
    assert 'metadata' in result
    metadata_handler.et.get_metadata.assert_called_once_with(str(Path(test_image_path).resolve()))

def test_write_metadata(metadata_handler, test_image_path, sample_metadata):
    """Test writing metadata to an image"""
    # Mock successful metadata write
    metadata_handler.et.execute.return_value = (0, "1 image files updated", "")
    metadata_handler.et.get_metadata.return_value = [{
        'Title': 'Test',
        'Description': sample_metadata['description'],
        'Keywords': sample_metadata['keywords']
    }]
    
    result = metadata_handler.write_metadata(test_image_path, sample_metadata)
    assert result['success'] is True
    assert 'modified_path' in result

def test_metadata_error_handling(metadata_handler, test_image_path):
    """Test error handling in metadata operations"""
    metadata_handler.et.get_metadata.side_effect = Exception("Test error")
    
    result = metadata_handler.read_metadata(test_image_path)
    assert result['success'] is False
    assert 'error' in result
    assert "Test error" in result['error']

@pytest.mark.parametrize("operation,expected_calls", [
    ('read', 'get_metadata'),
    ('write', 'execute')
])
def test_batch_operations(metadata_handler, test_images_dir,
                         operation, expected_calls, sample_metadata):
    """Test different batch operations"""
    image_paths = [
        str(Path(test_images_dir) / 'test1.jpg'),
        str(Path(test_images_dir) / 'test2.jpg')
    ]
    
    # Mock successful operations
    metadata_handler.et.execute.return_value = (0, "1 image files updated", "")
    metadata_handler.et.get_metadata.return_value = [{
        'Title': 'Test',
        'Description': sample_metadata['description'],
        'Keywords': sample_metadata['keywords']
    }]

    if operation == 'read':
        for path in image_paths:
            result = metadata_handler.read_metadata(path)
            assert result['success'] is True
            assert 'metadata' in result
    else:
        for path in image_paths:
            result = metadata_handler.write_metadata(path, sample_metadata)
            assert result['success'] is True
            assert 'modified_path' in result 