import os
import pytest
import pandas as pd
from lxml import etree
from pathlib import Path
from typing import Dict, Any, List

from src.core.metadata_handler import MetadataHandler

@pytest.fixture
def sample_results() -> List[Dict[str, Any]]:
    """Sample structured results for testing."""
    return [
        {
            'path': '/path/to/image1.jpg',
            'success': True,
            'description': 'A beautiful sunset',
            'keywords': ['sunset', 'nature', 'orange'],
            'technical_details': {
                'format': 'JPEG',
                'dimensions': '1920x1080',
                'color_space': 'sRGB'
            },
            'visual_elements': ['sun', 'clouds', 'horizon'],
            'composition': ['rule of thirds', 'leading lines'],
            'mood': 'peaceful',
            'use_cases': ['wallpaper', 'nature photography']
        },
        {
            'path': '/path/to/image2.jpg',
            'success': True,
            'description': 'A mountain landscape',
            'keywords': ['mountain', 'landscape', 'snow'],
            'technical_details': {
                'format': 'JPEG',
                'dimensions': '3840x2160',
                'color_space': 'sRGB'
            },
            'visual_elements': ['peaks', 'snow', 'sky'],
            'composition': ['symmetry', 'depth'],
            'mood': 'majestic',
            'use_cases': ['travel photography', 'prints']
        }
    ]

def test_save_to_csv(tmp_path, sample_results):
    """Test saving structured output to CSV."""
    handler = MetadataHandler()
    output_path = tmp_path / 'results.csv'
    
    # Process and save results
    handler.process_batch(
        image_paths=[r['path'] for r in sample_results],
        operation='read',
        output_format='csv',
        output_path=str(output_path)
    )
    
    # Verify CSV file was created
    assert output_path.exists()
    
    # Read and verify contents
    df = pd.read_csv(output_path)
    assert len(df) == 2
    
    # Check columns
    expected_columns = {
        'path', 'success', 'description', 'keywords',
        'technical_format', 'technical_dimensions', 'technical_color_space',
        'visual_elements', 'composition', 'mood', 'use_cases'
    }
    assert set(df.columns) == expected_columns
    
    # Check values
    assert df.iloc[0]['description'] == 'A beautiful sunset'
    assert 'sunset,nature,orange' in df.iloc[0]['keywords']
    assert df.iloc[1]['description'] == 'A mountain landscape'
    assert 'mountain,landscape,snow' in df.iloc[1]['keywords']

def test_save_to_xml(tmp_path, sample_results):
    """Test saving structured output to XML."""
    handler = MetadataHandler()
    output_path = tmp_path / 'results.xml'
    
    # Process and save results
    handler.process_batch(
        image_paths=[r['path'] for r in sample_results],
        operation='read',
        output_format='xml',
        output_path=str(output_path)
    )
    
    # Verify XML file was created
    assert output_path.exists()
    
    # Parse and verify XML structure
    tree = etree.parse(str(output_path))
    root = tree.getroot()
    
    # Check number of images
    images = root.findall('image')
    assert len(images) == 2
    
    # Check first image
    image1 = images[0]
    assert image1.find('path').text == '/path/to/image1.jpg'
    assert image1.find('description').text == 'A beautiful sunset'
    
    # Check keywords
    keywords = image1.find('keywords').findall('keyword')
    assert len(keywords) == 3
    assert {k.text for k in keywords} == {'sunset', 'nature', 'orange'}
    
    # Check technical details
    tech = image1.find('technical_details')
    assert tech.find('format').text == 'JPEG'
    assert tech.find('dimensions').text == '1920x1080'
    
    # Check second image
    image2 = images[1]
    assert image2.find('path').text == '/path/to/image2.jpg'
    assert image2.find('description').text == 'A mountain landscape'
    
    # Check mood and use cases
    assert image2.find('mood').text == 'majestic'
    use_cases = image2.find('use_cases').findall('item')
    assert len(use_cases) == 2
    assert {uc.text for uc in use_cases} == {'travel photography', 'prints'} 