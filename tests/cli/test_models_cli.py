"""
CLI test script for verifying model functionality
"""

import os
import pytest
from pathlib import Path
from click.testing import CliRunner
from src.cli import cli
from unittest.mock import MagicMock, AsyncMock
from unittest.mock import patch
from pytest_mock import mocker

@pytest.fixture
def runner():
    """Create a Click CLI runner"""
    return CliRunner()

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv('GOOGLE_API_KEY', 'test_key')
    monkeypatch.setenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')

@pytest.fixture
def mock_image_processor(mocker):
    """Mock ImageProcessor for testing"""
    processor = mocker.patch('src.cli.main.ImageProcessor')
    processor_instance = mocker.MagicMock()
    processor.return_value = processor_instance
    processor_instance.process_single.return_value = {
        'success': True,
        'original_path': 'test.jpg',
        'original_filename': 'test.jpg',
        'description': 'Test image',
        'keywords': ['test'],
        'technical_details': 'JPEG image',
        'visual_elements': ['test element'],
        'composition': 'test composition',
        'mood': 'test mood',
        'use_cases': ['test use case'],
        'suggested_filename': 'test_suggested.jpg'
    }
    return processor_instance

def test_single_gemini_csv(runner, test_image_path, mock_image_processor, mock_env):
    """Test single image processing with Gemini model and CSV output"""
    with runner.isolated_filesystem():
        with patch('src.core.image_processor.get_provider') as mock_provider:
            result = runner.invoke(cli, [
                'process',
                test_image_path,
                '--output-format', 'csv'
            ])
            assert result.exit_code == 0
            assert mock_image_processor.analyze_image.called

def test_single_gemini_xml(runner, test_image_path, mock_image_processor, mock_env):
    """Test single image processing with Gemini model and XML output"""
    with runner.isolated_filesystem():
        with patch('src.core.image_processor.get_provider') as mock_provider:
            result = runner.invoke(cli, [
                'process',
                test_image_path,
                '--output-format', 'xml'
            ])
            assert result.exit_code == 0
            assert mock_image_processor.analyze_image.called

@pytest.mark.parametrize("model", [
    '2-flash',
    '1.5-flash',
    '1.5-pro',
    'pro'
])
def test_model_selection(runner, test_image_path, mock_image_processor, mock_env, model):
    """Test different model selections"""
    with runner.isolated_filesystem():
        with patch('src.core.image_processor.get_provider') as mock_provider:
            result = runner.invoke(cli, [
                'process',
                test_image_path,
                '--model', model
            ])
            assert result.exit_code == 0
            assert mock_image_processor.analyze_image.called

def test_batch_processing(runner, test_images_dir, mock_image_processor, mock_env):
    """Test batch processing of images"""
    with runner.isolated_filesystem():
        result = runner.invoke(cli, [
            'process',
            test_images_dir,
            '--batch-size', '5',
            '--output-format', 'csv'
        ])
        assert result.exit_code == 0
        assert "Processing complete" in result.output

def test_compression_flag(runner, test_image_path, mock_image_processor, mock_env):
    """Test image compression flag"""
    with runner.isolated_filesystem():
        result = runner.invoke(cli, [
            'process',
            test_image_path,
            '--compress'
        ])
        assert result.exit_code == 0
        assert "Processing complete" in result.output

def test_recursive_processing(runner, test_images_dir, mock_image_processor, mock_env):
    """Test recursive directory processing"""
    with runner.isolated_filesystem():
        result = runner.invoke(cli, [
            'process',
            test_images_dir,
            '--recursive'
        ])
        assert result.exit_code == 0
        assert "Processing complete" in result.output

def test_prefix_option(runner, test_image_path, mock_image_processor, mock_env):
    """Test custom prefix for output files"""
    prefix = "test_prefix_"
    with runner.isolated_filesystem():
        result = runner.invoke(cli, [
            'process',
            test_image_path,
            '--prefix', prefix
        ])
        assert result.exit_code == 0
        assert "Processing complete" in result.output

def test_help_command(runner):
    """Test help command output"""
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert "Usage:" in result.output

def test_version_command(runner):
    """Test version command"""
    result = runner.invoke(cli, ['--version'])
    assert result.exit_code == 0
    assert "version" in result.output.lower()

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

def test_error_handling_invalid_path(runner):
    """Test error handling for invalid file path"""
    result = runner.invoke(cli, [
        'process',
        'nonexistent_file.jpg'
    ])
    assert result.exit_code == 2
    assert "does not exist" in result.output

def test_missing_api_key(runner, test_image_path, monkeypatch):
    """Test error handling when API key is missing"""
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    result = runner.invoke(cli, [
        'process',
        test_image_path
    ])
    assert result.exit_code == 1
    assert "GOOGLE_API_KEY environment variable not set" in result.output