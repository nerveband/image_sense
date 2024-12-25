"""
CLI test script for verifying model functionality
"""

import os
import pytest
from pathlib import Path
from click.testing import CliRunner
from src.cli import cli

@pytest.fixture
def runner():
    return CliRunner()

def test_single_gemini_csv(runner, test_image_path):
    """Test single image processing with Gemini model and CSV output"""
    result = runner.invoke(cli, [
        'process',
        test_image_path,
        '--output-format', 'csv'
    ])
    assert result.exit_code == 0
    assert "Metadata saved to:" in result.output

def test_single_gemini_xml(runner, test_image_path):
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