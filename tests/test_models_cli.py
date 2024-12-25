"""
CLI test script for verifying model functionality
"""

import os
import pytest
from click.testing import CliRunner
from src.cli import process_images  # Assuming your CLI entry point is here
import pandas as pd
from lxml import etree

# Constants for test paths
TEST_IMAGES_DIR = "tests/test_images"
TEST_METADATA_DIR = "tests/test_metadata"
TEST_IMAGE_PATH = os.path.join(TEST_IMAGES_DIR, "test.jpg")
TEST_CSV_PATH = os.path.join(TEST_METADATA_DIR, "metadata.csv")
TEST_XML_PATH = os.path.join(TEST_METADATA_DIR, "metadata.xml")

# Fixture to set up test environment
@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    # Create test directories and a dummy image if they don't exist
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(TEST_METADATA_DIR, exist_ok=True)
    if not os.path.exists(TEST_IMAGE_PATH):
        # Create a dummy image file (replace with actual image creation if needed)
        open(TEST_IMAGE_PATH, "w").close()
    yield
    # Cleanup: Remove generated files after tests
    if os.path.exists(TEST_CSV_PATH):
        os.remove(TEST_CSV_PATH)
    if os.path.exists(TEST_XML_PATH):
        os.remove(TEST_XML_PATH)

# Test single image processing with Gemini model and CSV output
def test_single_gemini_csv():
    runner = CliRunner()
    result = runner.invoke(
        process_images,
        [
            TEST_IMAGE_PATH,
            "--model",
            "gemini",
            "--output",
            "csv",
            "--output-file",
            TEST_CSV_PATH,
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(TEST_CSV_PATH)
    df = pd.read_csv(TEST_CSV_PATH)
    assert not df.empty
    # Add more assertions to validate CSV content

# Test single image processing with Anthropic model and CSV output
def test_single_anthropic_csv():
    runner = CliRunner()
    result = runner.invoke(
        process_images,
        [
            TEST_IMAGE_PATH,
            "--model",
            "anthropic",
            "--output",
            "csv",
            "--output-file",
            TEST_CSV_PATH,
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(TEST_CSV_PATH)
    df = pd.read_csv(TEST_CSV_PATH)
    assert not df.empty
    # Add more assertions to validate CSV content

# Test single image processing with Gemini model and XML output
def test_single_gemini_xml():
    runner = CliRunner()
    result = runner.invoke(
        process_images,
        [
            TEST_IMAGE_PATH,
            "--model",
            "gemini",
            "--output",
            "xml",
            "--output-file",
            TEST_XML_PATH,
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(TEST_XML_PATH)
    tree = etree.parse(TEST_XML_PATH)
    assert tree is not None
    # Add more assertions to validate XML content

# Test single image processing with Anthropic model and XML output
def test_single_anthropic_xml():
    runner = CliRunner()
    result = runner.invoke(
        process_images,
        [
            TEST_IMAGE_PATH,
            "--model",
            "anthropic",
            "--output",
            "xml",
            "--output-file",
            TEST_XML_PATH,
        ],
    )
    assert result.exit_code == 0
    assert os.path.exists(TEST_XML_PATH)
    tree = etree.parse(TEST_XML_PATH)
    assert tree is not None
    # Add more assertions to validate XML content 