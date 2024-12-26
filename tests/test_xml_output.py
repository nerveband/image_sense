import os
import pytest
import tempfile
from PIL import Image
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from lxml import etree
from src.core.image_processor import ImageProcessor

@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<image_analysis>
    <description>A beautiful sunset over mountains with orange and purple sky</description>
    <keywords>
        <keyword>sunset</keyword>
        <keyword>mountains</keyword>
        <keyword>orange sky</keyword>
        <keyword>purple sky</keyword>
        <keyword>nature</keyword>
    </keywords>
    <technical_details>High contrast scene with warm and cool tones</technical_details>
    <visual_elements>Mountains silhouetted against colorful sky</visual_elements>
    <composition>Rule of thirds with mountains in lower third</composition>
    <mood>Peaceful and serene</mood>
    <use_cases>Nature photography, wallpapers</use_cases>
    <suggested_filename>mountain_sunset.jpg</suggested_filename>
</image_analysis>"""

@pytest.fixture
def test_image_path():
    """Create a test image file"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, "test.jpg")
        # Create a small test image using PIL
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(image_path, format='JPEG')
        yield image_path

@pytest.fixture
def mock_gemini_provider(mock_gemini_response):
    """Mock Gemini provider"""
    provider = AsyncMock()
    provider.analyze_image.return_value = mock_gemini_response
    provider.analyze_batch.return_value = [mock_gemini_response]
    return provider

@pytest.mark.asyncio
async def test_xml_parsing(test_image_path, mock_gemini_provider):
    """Test XML response parsing"""
    with patch('src.core.image_processor.get_provider', return_value=mock_gemini_provider):
        processor = ImageProcessor(api_key="dummy_key")
        results = await processor.process_batch([test_image_path])

        assert len(results) == 1
        result = results[0]

        # Verify parsed fields
        assert result['description'] == "A beautiful sunset over mountains with orange and purple sky"
        assert "sunset" in result['keywords']
        assert "mountains" in result['keywords']
        assert result['technical_details'] == "High contrast scene with warm and cool tones"
        assert result['visual_elements'] == "Mountains silhouetted against colorful sky"
        assert result['composition'] == "Rule of thirds with mountains in lower third"
        assert result['mood'] == "Peaceful and serene"
        assert result['use_cases'] == "Nature photography, wallpapers"
        assert result['suggested_filename'] == "mountain_sunset.jpg"

@pytest.mark.asyncio
async def test_xml_output(test_image_path, mock_gemini_provider, tmp_path):
    """Test XML file output"""
    output_path = tmp_path / "results.xml"

    with patch('src.core.image_processor.get_provider', return_value=mock_gemini_provider):
        processor = ImageProcessor(api_key="dummy_key")
        results = await processor.process_batch([test_image_path])

        # Save results to XML
        processor.save_to_xml(results, str(output_path))

        # Verify XML file exists
        assert output_path.exists()

        # Parse and verify XML structure
        tree = etree.parse(str(output_path))
        root = tree.getroot()

        assert root.tag == "image_analysis_results"
        image = root.find("image")
        assert image is not None

        # Verify XML elements
        assert image.find(".//description").text == "A beautiful sunset over mountains with orange and purple sky"
        assert len(image.findall(".//keyword")) >= 1
        assert image.find(".//technical_details").text == "High contrast scene with warm and cool tones"
        assert image.find(".//visual_elements").text == "Mountains silhouetted against colorful sky"
        assert image.find(".//composition").text == "Rule of thirds with mountains in lower third"
        assert image.find(".//mood").text == "Peaceful and serene"
        assert image.find(".//use_cases").text == "Nature photography, wallpapers"
        assert image.find(".//suggested_filename").text == "mountain_sunset.jpg"

@pytest.mark.asyncio
async def test_metadata_writing(test_image_path, mock_gemini_provider):
    """Test metadata writing from XML results"""
    with patch('src.core.image_processor.get_provider', return_value=mock_gemini_provider):
        processor = ImageProcessor(api_key="dummy_key")
        results = await processor.process_batch([test_image_path])

        assert len(results) == 1
        result = results[0]

        # Verify metadata was written
        assert result['success'] is True
        assert 'description' in result
        assert 'keywords' in result
        assert 'technical_details' in result

@pytest.mark.asyncio
async def test_batch_processing(mock_gemini_provider, tmp_path):
    """Test batch processing with XML output"""
    # Create multiple test images
    image_paths = []
    for i in range(3):
        image_path = tmp_path / f"test_{i}.jpg"
        # Create a small test image using PIL
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(str(image_path), format='JPEG')
        image_paths.append(str(image_path))

    output_path = tmp_path / "results.xml"

    with patch('src.core.image_processor.get_provider', return_value=mock_gemini_provider):
        processor = ImageProcessor(api_key="dummy_key")
        results = await processor.process_batch(image_paths)

        # Save results to XML
        processor.save_to_xml(results, str(output_path))

        # Verify results
        assert len(results) == 3
        for result in results:
            assert result['success'] is True
            assert 'description' in result
            assert 'keywords' in result
            assert 'technical_details' in result 