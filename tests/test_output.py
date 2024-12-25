"""
Test output handler functionality for CSV and XML formats.
"""

import os
import pytest
from pathlib import Path
from src.core.output_handler import OutputHandler

@pytest.fixture
def test_data():
    """Sample metadata for testing"""
    return [
        {
            'file': 'test1.jpg',
            'description': 'A test image',
            'keywords': ['test', 'image'],
            'categories': ['test']
        },
        {
            'file': 'test2.jpg',
            'description': 'Another test image',
            'keywords': ['test', 'another'],
            'categories': ['test', 'sample']
        }
    ]

def test_csv_export(test_data, temp_output_dir):
    """Test CSV export functionality"""
    handler = OutputHandler(temp_output_dir)
    output_file = 'test_output.csv'
    
    # Export to CSV
    handler.export_to_csv(test_data, output_file)
    
    # Verify the file exists
    output_path = temp_output_dir / output_file
    assert output_path.exists()
    
    # Read and verify contents
    with open(output_path, 'r') as f:
        lines = f.readlines()
        
    # Check header
    header = lines[0].strip().split(',')
    expected_fields = {'file', 'description', 'keywords', 'categories'}
    assert set(header) == expected_fields
    
    # Check data
    assert len(lines) == 3  # Header + 2 data rows
    
    # Verify first row data is present
    first_row = lines[1]
    assert 'test1.jpg' in first_row
    assert 'A test image' in first_row
    assert 'test,image' in first_row
    
    # Verify second row data is present
    second_row = lines[2]
    assert 'test2.jpg' in second_row
    assert 'Another test image' in second_row
    assert 'test,another' in second_row

def test_xml_export(test_data, temp_output_dir):
    """Test XML export functionality"""
    handler = OutputHandler(temp_output_dir)
    output_file = 'test_output.xml'
    
    # Export to XML
    handler.export_to_xml(test_data, output_file)
    
    # Verify the file exists
    output_path = temp_output_dir / output_file
    assert output_path.exists()
    
    # Read and verify contents
    import xml.etree.ElementTree as ET
    tree = ET.parse(output_path)
    root = tree.getroot()
    
    # Check root element
    assert root.tag == 'metadata'
    
    # Check number of images
    images = root.findall('image')
    assert len(images) == 2
    
    # Check first image
    first_image = images[0]
    assert first_image.find('file').text == 'test1.jpg'
    assert first_image.find('description').text == 'A test image'
    assert 'test' in first_image.find('keywords').text
    assert 'image' in first_image.find('keywords').text
    
    # Check second image
    second_image = images[1]
    assert second_image.find('file').text == 'test2.jpg'
    assert second_image.find('description').text == 'Another test image'
    assert 'test' in second_image.find('keywords').text
    assert 'another' in second_image.find('keywords').text
    assert 'test' in second_image.find('categories').text
    assert 'sample' in second_image.find('categories').text

def test_output_directory_creation(temp_output_dir):
    """Test that the output directory is created if it doesn't exist"""
    new_dir = temp_output_dir / 'new_output_dir'
    handler = OutputHandler(new_dir)
    
    assert new_dir.exists()

def test_invalid_data_handling(temp_output_dir):
    """Test handling of invalid data"""
    handler = OutputHandler(temp_output_dir)
    
    # Test with empty data
    handler.export_to_csv([], 'empty.csv')
    handler.export_to_xml([], 'empty.xml')
    
    # Test with None values
    data_with_none = [{'file': 'test.jpg', 'description': None}]
    handler.export_to_csv(data_with_none, 'none_values.csv')
    handler.export_to_xml(data_with_none, 'none_values.xml')
    
    # Files should be created without errors
    assert (temp_output_dir / 'empty.csv').exists()
    assert (temp_output_dir / 'empty.xml').exists()
    assert (temp_output_dir / 'none_values.csv').exists()
    assert (temp_output_dir / 'none_values.xml').exists()

def test_special_characters(temp_output_dir):
    """Test handling of special characters in metadata"""
    data = [{
        'file': 'test.jpg',
        'description': 'Special chars: &<>"\',',
        'keywords': ['test, with, commas', 'xml & chars'],
        'categories': ['test & category']
    }]
    
    handler = OutputHandler(temp_output_dir)
    
    # Test CSV export
    handler.export_to_csv(data, 'special_chars.csv')
    csv_path = temp_output_dir / 'special_chars.csv'
    assert csv_path.exists()
    
    # Test XML export
    handler.export_to_xml(data, 'special_chars.xml')
    xml_path = temp_output_dir / 'special_chars.xml'
    assert xml_path.exists()
    
    # Verify XML can be parsed
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    assert root is not None

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 