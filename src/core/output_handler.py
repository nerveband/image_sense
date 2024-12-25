"""Output handling utilities"""

import os
import csv
import json
from pathlib import Path
from typing import Dict, Any
import xml.etree.ElementTree as ET
from xml.dom import minidom

def save_metadata(metadata: str, image_path: str, output_format: str = 'csv') -> str:
    """
    Save metadata to a file in the specified format
    
    Args:
        metadata: The metadata to save
        image_path: Path to the original image
        output_format: Format to save in (csv or xml)
        
    Returns:
        Path to the saved metadata file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(image_path), 'metadata')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    if output_format == 'csv':
        output_path = os.path.join(output_dir, f"{base_name}_metadata.csv")
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Metadata'])
            writer.writerow([os.path.basename(image_path), metadata])
    else:  # xml
        output_path = os.path.join(output_dir, f"{base_name}_metadata.xml")
        root = ET.Element('metadata')
        image = ET.SubElement(root, 'image')
        image.text = os.path.basename(image_path)
        description = ET.SubElement(root, 'description')
        description.text = metadata
        
        # Pretty print XML
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        with open(output_path, 'w') as f:
            f.write(xml_str)
            
    return output_path 