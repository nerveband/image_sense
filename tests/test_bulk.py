"""
Test bulk image processing functionality
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from click.testing import CliRunner
from src.cli.main import cli
import numpy as np
from PIL import Image

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture(autouse=True)
def mock_image_processor():
    """Mock ImageProcessor to prevent real API calls"""
    with patch('src.cli.main.ImageProcessor') as mock:
        # Create a mock instance
        instance = mock.return_value
        
        # Mock the async methods
        instance.analyze_image = AsyncMock(return_value={
            'success': True,
            'description': 'Test image',
            'keywords': ['test'],
            'technical_details': {
                'format': 'JPEG',
                'dimensions': '100x100'
            },
            'visual_elements': ['element1'],
            'composition': ['composition1'],
            'mood': 'test mood',
            'use_cases': ['use case 1'],
            'suggested_filename': 'test.jpg'
        })
        
        instance.process_images = AsyncMock(return_value=[{
            'success': True,
            'description': 'Test image',
            'keywords': ['test'],
            'technical_details': {
                'format': 'JPEG',
                'dimensions': '100x100'
            },
            'visual_elements': ['element1'],
            'composition': ['composition1'],
            'mood': 'test mood',
            'use_cases': ['use case 1'],
            'suggested_filename': 'test.jpg'
        }])
        
        yield instance

@pytest.fixture
def test_images_dir(tmp_path):
    """Create a temporary directory with test images"""
    # Create main directory
    img_dir = tmp_path / "test_images"
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
    
    return str(img_dir)

def test_bulk_gemini_csv(runner, test_images_dir, mock_image_processor):
    """Test bulk processing with Gemini model and CSV output"""
    result = runner.invoke(cli, [
        'bulk-process',
        test_images_dir,
        '--output-format', 'csv'
    ])

    assert result.exit_code == 0
    assert "Processing 2 images" in result.output
    assert "Results saved to" in result.output

def test_bulk_gemini_xml(runner, test_images_dir, mock_image_processor):
    """Test bulk processing with Gemini model and XML output"""
    result = runner.invoke(cli, [
        'bulk-process',
        test_images_dir,
        '--output-format', 'xml'
    ])

    assert result.exit_code == 0
    assert "Processing 2 images" in result.output
    assert "Results saved to" in result.output

def test_bulk_recursive(runner, test_images_dir, mock_image_processor):
    """Test recursive bulk processing"""
    result = runner.invoke(cli, [
        'bulk-process',
        test_images_dir,
        '--recursive',
        '--output-format', 'csv'
    ])

    assert result.exit_code == 0
    assert "Processing directory recursively" in result.output
    assert "Processing 4 images" in result.output
    assert "Results saved to" in result.output

def test_bulk_empty_directory(runner, tmp_path):
    """Test bulk processing with empty directory"""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = runner.invoke(cli, [
        'bulk-process',
        str(empty_dir)
    ])

    assert result.exit_code == 1
    assert "No images found in directory" in result.output

def test_bulk_invalid_directory(runner):
    """Test bulk processing with non-existent directory"""
    result = runner.invoke(cli, [
        'bulk-process',
        '/nonexistent/directory'
    ])

    assert result.exit_code == 2
    assert "Error:" in result.output

def test_bulk_with_compression(runner, test_images_dir, mock_image_processor):
    """Test bulk processing with image compression"""
    result = runner.invoke(cli, [
        'bulk-process',
        test_images_dir,
        '--compress',
        '--output-format', 'csv'
    ])

    assert result.exit_code == 0
    assert "Processing 2 images" in result.output
    assert "Results saved to" in result.output

if __name__ == "__main__":
    pytest.main(["-v", __file__])