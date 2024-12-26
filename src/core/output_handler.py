"""Output handler module."""
import os
import csv
import xml.etree.ElementTree as ET
from io import StringIO
from typing import List, Dict, Any
import logging

def save_metadata(result: Dict[str, Any], image_path: str, output_format: str = 'csv') -> str:
    """Save metadata to a file in the specified format."""
    # Create output path
    output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_metadata.{output_format}")
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if output_format == 'csv':
            # Save as CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=result.keys())
                writer.writeheader()
                # Convert lists to strings
                row = result.copy()
                for key, value in row.items():
                    if isinstance(value, (list, tuple)):
                        row[key] = '; '.join(str(v) for v in value)
                writer.writerow(row)
        else:
            # Save as XML
            import xml.etree.ElementTree as ET
            root = ET.Element("image_metadata")
            for key, value in result.items():
                elem = ET.SubElement(root, key)
                if isinstance(value, (list, tuple)):
                    for item in value:
                        item_elem = ET.SubElement(elem, "item")
                        item_elem.text = str(item)
                else:
                    elem.text = str(value)
            tree = ET.ElementTree(root)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        return output_path
        
    except Exception as e:
        logging.error(f"Error saving metadata: {str(e)}")
        raise

class OutputHandler:
    """Handler for output operations."""
    
    def __init__(self):
        """Initialize the output handler."""
        self.logger = logging.getLogger(__name__)
        
    def save_to_csv(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """Save results to CSV file."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Determine fields from first result
            if not results:
                return
                
            fields = ['original_path', 'success', 'error', 'description', 'keywords', 
                     'technical_details', 'visual_elements', 'composition', 'mood', 
                     'use_cases', 'suggested_filename']
            
            def parse_xml_content(content: str) -> Dict[str, Any]:
                """Parse XML content into a dictionary.
                
                Args:
                    content: XML string to parse
                    
                Returns:
                    Dictionary containing parsed XML data
                """
                try:
                    self.logger.debug(f"Attempting to parse XML content: {content[:200]}...")
                    
                    if not content.strip().startswith('<?xml'):
                        self.logger.debug("Content is not XML, returning as description")
                        return {'description': content}
                    
                    # Parse XML string
                    root = ET.fromstring(content)
                    result = {}
                    
                    # Extract text from each element
                    for child in root:
                        try:
                            if child.tag == 'technical_details':
                                # Handle technical details specially
                                tech_details = {}
                                for detail in child:
                                    if detail.text:
                                        tech_details[detail.tag] = detail.text.strip()
                                result[child.tag] = tech_details
                            elif len(child) > 0:  # Has sub-elements
                                # For lists like keywords, visual_elements etc.
                                items = []
                                for item in child:
                                    if item.text:
                                        items.append(item.text.strip())
                                    else:
                                        self.logger.warning(f"Empty item found in {child.tag}")
                                result[child.tag] = items
                            else:
                                if child.text:
                                    result[child.tag] = child.text.strip()
                                else:
                                    self.logger.warning(f"Empty element found: {child.tag}")
                        except Exception as e:
                            self.logger.error(f"Error parsing element {child.tag}: {str(e)}")
                            # Continue with other elements
                            continue
                    
                    # Validate required fields
                    required_fields = {'description', 'keywords', 'technical_details', 
                                     'visual_elements', 'composition', 'mood', 
                                     'use_cases', 'suggested_filename'}
                    missing_fields = required_fields - set(result.keys())
                    if missing_fields:
                        self.logger.warning(f"Missing required fields in XML: {missing_fields}")
                    
                    return result
                    
                except ET.ParseError as e:
                    self.logger.error(f"XML parsing error: {str(e)}")
                    return {'description': content, 'error': f"XML parsing error: {str(e)}"}
                except Exception as e:
                    self.logger.error(f"Unexpected error parsing XML: {str(e)}")
                    return {'description': content, 'error': f"Error parsing content: {str(e)}"}
            
            # Write results to CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                for result in results:
                    # Create a new dictionary for the row to avoid modifying the original
                    row = {}
                    # First copy all non-XML content
                    for key, value in result.items():
                        if isinstance(value, (list, tuple)):
                            row[key] = '; '.join(str(v) for v in value)
                        else:
                            row[key] = value
                    
                    # Then handle XML content separately
                    for key, value in result.items():
                        if isinstance(value, str) and '<?xml' in value:
                            # Parse XML content
                            parsed = parse_xml_content(value)
                            # Update row with parsed values
                            for parsed_key, parsed_value in parsed.items():
                                if isinstance(parsed_value, dict):
                                    # Handle nested dictionaries (like technical_details)
                                    row[parsed_key] = str(parsed_value)
                                elif isinstance(parsed_value, (list, tuple)):
                                    row[parsed_key] = '; '.join(str(v) for v in parsed_value)
                                else:
                                    row[parsed_key] = parsed_value
                    
                    writer.writerow(row)
                    
            self.logger.info(f"Saved results to CSV: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving CSV: {str(e)}")
            raise 