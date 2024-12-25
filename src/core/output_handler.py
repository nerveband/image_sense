"""
Output handler for saving metadata in different formats
"""

import os
import csv
from pathlib import Path
from typing import List, Dict, Any
from lxml import etree

class OutputHandler:
    """Handles saving metadata in different formats"""
    
    def __init__(self, output_dir: str):
        """
        Initialize the output handler
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_to_csv(self, metadata: List[Dict[str, Any]], filename: str):
        """
        Export metadata to CSV format
        
        Args:
            metadata: List of metadata dictionaries
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        # Ensure all required fields are present
        fieldnames = ['file', 'description', 'keywords', 'categories']
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in metadata:
                # Convert lists to comma-separated strings and handle missing fields
                row = {
                    'file': item.get('file', ''),
                    'description': item.get('description', ''),
                    'keywords': ','.join(item.get('keywords', [])) if isinstance(item.get('keywords'), list) else item.get('keywords', ''),
                    'categories': ','.join(item.get('categories', [])) if isinstance(item.get('categories'), list) else item.get('categories', '')
                }
                writer.writerow(row)
    
    def export_to_xml(self, metadata: List[Dict[str, Any]], filename: str):
        """
        Export metadata to XML format
        
        Args:
            metadata: List of metadata dictionaries
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        # Create root element
        root = etree.Element('metadata')
        
        for item in metadata:
            # Create image element
            image = etree.SubElement(root, 'image')
            
            # Add file element
            file_elem = etree.SubElement(image, 'file')
            file_elem.text = item['file']
            
            # Add description element
            desc_elem = etree.SubElement(image, 'description')
            desc_elem.text = item['description']
            
            # Add keywords element
            keywords_elem = etree.SubElement(image, 'keywords')
            keywords_elem.text = ','.join(item['keywords']) if isinstance(item['keywords'], list) else item['keywords']
            
            # Add categories element
            categories_elem = etree.SubElement(image, 'categories')
            categories_elem.text = ','.join(item['categories']) if isinstance(item['categories'], list) else item['categories']
        
        # Write to file
        tree = etree.ElementTree(root)
        tree.write(str(output_path), pretty_print=True, xml_declaration=True, encoding='utf-8') 