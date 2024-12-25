"""
Test bulk image processing functionality
"""

import os
import pytest
from pathlib import Path
from click.testing import CliRunner
from src.cli.main import cli

@pytest.fixture
def runner():
    return CliRunner()

def test_bulk_gemini_csv(runner, test_images_dir):
    """Test bulk processing with Gemini model and CSV output"""
    result = runner.invoke(cli, [
        'bulk-process',
        test_images_dir,
        '--output-format', 'csv'
    ])
    assert result.exit_code == 0
    assert "Processing images" in result.output

def test_bulk_gemini_xml(runner, test_images_dir):
    """Test bulk processing with Gemini model and XML output"""
    result = runner.invoke(cli, [
        'bulk-process',
        test_images_dir,
        '--output-format', 'xml'
    ])
    assert result.exit_code == 0
    assert "Processing images" in result.output

def test_bulk_recursive(runner, test_images_dir):
    """Test recursive bulk processing"""
    result = runner.invoke(cli, [
        'bulk-process',
        test_images_dir,
        '--recursive'
    ])
    assert result.exit_code == 0
    assert "Processing images" in result.output

def test_bulk_empty_directory(runner, tmp_path):
    """Test bulk processing with empty directory"""
    result = runner.invoke(cli, [
        'bulk-process',
        str(tmp_path)
    ])
    assert result.exit_code == 1
    assert "No images found in directory" in result.output

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 