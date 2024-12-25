"""
CLI test script for verifying model functionality
"""

import os
import pytest
from pathlib import Path
from click.testing import CliRunner
from src.cli import cli
from unittest.mock import MagicMock, patch

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def mock_gemini_provider(mocker):
    """Mock the Gemini provider"""
    mock_provider = MagicMock()
    mock_provider.analyze_image.return_value = "Test metadata"
    mocker.patch('src.core.llm_handler.GeminiProvider', return_value=mock_provider)
    return mock_provider

@pytest.fixture
def mock_image_processor(mocker):
    """Mock the image processor"""
    mock_processor = MagicMock()
    mock_processor.process_images.return_value = [{"success": True, "description": "Test"}]
    mocker.patch('src.cli.ImageProcessor', return_value=mock_processor)
    return mock_processor

def test_single_gemini_csv(runner, test_image_path, mock_gemini_provider):
    """Test single image processing with Gemini model and CSV output"""
    result = runner.invoke(cli, [
        'process',
        test_image_path,
        '--output-format', 'csv'
    ])
    assert result.exit_code == 0
    assert "Metadata saved to:" in result.output

def test_single_gemini_xml(runner, test_image_path, mock_gemini_provider):
    """Test single image processing with Gemini model and XML output"""
    result = runner.invoke(cli, [
        'process',
        test_image_path,
        '--output-format', 'xml'
    ])
    assert result.exit_code == 0
    assert "Metadata saved to:" in result.output

def test_missing_api_key(runner, test_image_path, monkeypatch):
    """Test error handling when API key is missing"""
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    result = runner.invoke(cli, [
        'process',
        test_image_path
    ])
    assert result.exit_code == 1
    assert "GOOGLE_API_KEY environment variable not set" in result.output

def test_batch_processing(runner, test_images_dir, mock_image_processor):
    """Test batch processing of images"""
    result = runner.invoke(cli, [
        'process',
        test_images_dir,
        '--batch-size', '5',
        '--output-format', 'csv'
    ])
    assert result.exit_code == 0
    assert "Processing directory:" in result.output
    mock_image_processor.process_images.assert_called_once()

def test_invalid_output_format(runner, test_image_path):
    """Test error handling for invalid output format"""
    result = runner.invoke(cli, [
        'process',
        test_image_path,
        '--output-format', 'invalid'
    ])
    assert result.exit_code == 2
    assert "Invalid value for '--output-format'" in result.output

def test_invalid_model(runner, test_image_path):
    """Test error handling for invalid model selection"""
    result = runner.invoke(cli, [
        'process',
        test_image_path,
        '--model', 'invalid-model'
    ])
    assert result.exit_code == 2
    assert "Invalid value for '--model'" in result.output

def test_compression_flag(runner, test_image_path, mock_image_processor):
    """Test image compression flag"""
    result = runner.invoke(cli, [
        'process',
        test_image_path,
        '--compress'
    ])
    assert result.exit_code == 0
    # Get the kwargs from the last call
    _, kwargs = mock_image_processor.process_images.call_args
    # Update kwargs with compress=True
    expected_kwargs = {**kwargs, 'compress': True}
    mock_image_processor.process_images.assert_called_with(test_image_path, **expected_kwargs)

def test_recursive_processing(runner, test_images_dir, mock_image_processor):
    """Test recursive directory processing"""
    result = runner.invoke(cli, [
        'process',
        test_images_dir,
        '--recursive'
    ])
    assert result.exit_code == 0
    assert "Processing directory recursively" in result.output

def test_prefix_option(runner, test_image_path, mock_image_processor):
    """Test custom prefix for output files"""
    prefix = "test_prefix_"
    result = runner.invoke(cli, [
        'process',
        test_image_path,
        '--prefix', prefix
    ])
    assert result.exit_code == 0
    assert mock_image_processor.called_with(prefix=prefix)

def test_help_command(runner):
    """Test help command output"""
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Options:" in result.output

def test_version_command(runner):
    """Test version command"""
    result = runner.invoke(cli, ['--version'])
    assert result.exit_code == 0
    assert "version" in result.output.lower()

@pytest.mark.parametrize("model", [
    '2-flash',
    '1.5-flash',
    '1.5-pro',
    'pro'
])
def test_model_selection(runner, test_image_path, mock_image_processor, model):
    """Test different model selections"""
    result = runner.invoke(cli, [
        'process',
        test_image_path,
        '--model', model
    ])
    assert result.exit_code == 0
    assert mock_image_processor.called_with(model=model)

def test_error_handling_invalid_path(runner):
    """Test error handling for invalid file path"""
    result = runner.invoke(cli, [
        'process',
        'nonexistent_file.jpg'
    ])
    assert result.exit_code == 2
    assert "does not exist" in result.output 