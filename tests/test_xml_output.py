import os
import pytest
import tempfile
from PIL import Image
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from lxml import etree
from src.core.image_processor import ImageProcessor
import xml.etree.ElementTree as ET

@pytest.fixture
def mock_gemini_response():
    """Mock response from Gemini provider"""
    return {
        'content': """<?xml version="1.0" encoding="UTF-8"?>
<image_analysis>
    <description>A beautiful sunset over mountains with orange and purple sky</description>
    <keywords>
        <keyword>sunset</keyword>
        <keyword>mountains</keyword>
        <keyword>orange sky</keyword>
        <keyword>purple sky</keyword>
    </keywords>
    <technical_details>JPEG format, 1920x1080 resolution</technical_details>
    <visual_elements>Mountains silhouetted against colorful sky</visual_elements>
    <composition>Rule of thirds with mountains in lower third</composition>
    <mood>Peaceful and serene</mood>
    <use_cases>Nature photography, wallpapers</use_cases>
    <suggested_filename>mountain_sunset.jpg</suggested_filename>
</image_analysis>""",
        'metadata': {
            'description': "Image analysis result",
            'format': "JPEG",
            'dimensions': "1920x1080"
        }
    }

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
    
    class MockResponse:
        def __init__(self, text):
            self.text = text
            
        def resolve(self):
            pass
    
    async def mock_generate(*args, **kwargs):
        return MockResponse(str(mock_gemini_response['content']))
    
    provider.generate_content = mock_generate
    return provider

@pytest.mark.asyncio
async def test_xml_parsing(test_image_path, mock_gemini_provider):
    """Test XML response parsing"""
    with patch('src.core.image_processor.get_provider', return_value=mock_gemini_provider):
        processor = ImageProcessor(api_key="dummy_key")
        results = await processor.process_batch([test_image_path])
        
        assert len(results) == 1
        result = results[0]
        assert 'content' in result
        assert 'metadata' in result
        assert result['content'] == str(mock_gemini_provider.generate_content.return_value.text)

@pytest.mark.asyncio
async def test_xml_output(test_image_path, mock_gemini_provider, tmp_path):
    """Test XML file output"""
    output_path = tmp_path / "results.xml"
    
    with patch('src.core.image_processor.get_provider', return_value=mock_gemini_provider):
        processor = ImageProcessor(api_key="dummy_key")
        results = await processor.process_batch([test_image_path])
        processor.save_to_xml(results, output_path)
        
        assert output_path.exists()
        tree = ET.parse(output_path)
        root = tree.getroot()
        
        assert root.tag == 'results'
        assert len(root) == 1
        image_elem = root[0]
        assert image_elem.tag == 'image'
        assert 'path' in image_elem.attrib
        assert any(child.tag == 'content' for child in image_elem)
        assert any(child.tag == 'metadata' for child in image_elem)

@pytest.mark.asyncio
async def test_metadata_writing(test_image_path, mock_gemini_provider):
    """Test metadata writing from XML results"""
    with patch('src.core.image_processor.get_provider', return_value=mock_gemini_provider):
        processor = ImageProcessor(api_key="dummy_key")
        results = await processor.process_batch([test_image_path])
        
        assert len(results) == 1
        result = results[0]
        assert 'metadata' in result
        metadata = result['metadata']
        assert isinstance(metadata, dict)
        assert 'description' in metadata
        assert 'format' in metadata
        assert 'dimensions' in metadata

@pytest.mark.asyncio
async def test_batch_processing(mock_gemini_provider, tmp_path):
    """Test batch processing with XML output"""
    # Create test images
    image_paths = []
    for i in range(3):
        image_path = tmp_path / f"test_{i}.jpg"
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        img.save(image_path, format='JPEG')
        image_paths.append(str(image_path))
    
    with patch('src.core.image_processor.get_provider', return_value=mock_gemini_provider):
        processor = ImageProcessor(api_key="dummy_key")
        results = await processor.process_batch(image_paths)
        
        assert len(results) == 3
        for result in results:
            assert 'content' in result
            assert 'metadata' in result
            assert result['content'] == str(mock_gemini_provider.generate_content.return_value.text)