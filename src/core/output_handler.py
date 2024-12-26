"""Output handler module."""
import os
import csv
import xml.etree.ElementTree as ET
from io import StringIO
from typing import List, Dict, Any, Union
import logging
from lxml import etree
from pathlib import Path

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

    def save_to_xml(self, results: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
        """Save results to XML file.
        
        Args:
            results: List of results to save
            output_path: Path to save XML file to
        """
        try:
            # Create root element
            root = ET.Element('results')
            
            # Add results
            for result in results:
                image_elem = ET.SubElement(root, 'image')
                
                # Add path attribute
                if 'original_path' in result:
                    image_elem.set('path', result['original_path'])
                
                # Add content if present
                if 'content' in result:
                    content_elem = ET.SubElement(image_elem, 'content')
                    content_elem.text = str(result['content'])
                
                # Add metadata if present
                if 'metadata' in result:
                    metadata_elem = ET.SubElement(image_elem, 'metadata')
                    self._add_metadata_to_xml(metadata_elem, result['metadata'])
                
                # Add error if present
                if 'error' in result:
                    error_elem = ET.SubElement(image_elem, 'error')
                    error_elem.text = str(result['error'])
            
            # Create XML tree and save
            tree = ET.ElementTree(root)
            tree.write(str(output_path), encoding='utf-8', xml_declaration=True)
            
            self.logger.info(f"Saved results to XML: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results to XML: {str(e)}")
            raise

def parse_xml_response(response_text: str) -> Dict[str, Any]:
    """Parse XML response into a dictionary."""
    try:
        # Parse XML string
        root = etree.fromstring(response_text)
        
        result = {}
        
        # Extract basic fields
        description = root.find('description')
        if description is not None:
            result['description'] = description.text
        
        # Extract keywords
        keywords = root.find('keywords')
        if keywords is not None:
            result['keywords'] = [kw.text for kw in keywords.findall('keyword') if kw.text]
        
        # Extract technical details
        tech_details = root.find('technical_details')
        if tech_details is not None:
            result['technical_details'] = {
                'format': tech_details.find('format').text if tech_details.find('format') is not None else None,
                'dimensions': tech_details.find('dimensions').text if tech_details.find('dimensions') is not None else None,
                'color_space': tech_details.find('color_space').text if tech_details.find('color_space') is not None else None
            }
        
        # Extract visual elements
        visual = root.find('visual_elements')
        if visual is not None:
            result['visual_elements'] = [elem.text for elem in visual.findall('element') if elem.text]
        
        # Extract composition
        composition = root.find('composition')
        if composition is not None:
            result['composition'] = [comp.text for comp in composition.findall('technique') if comp.text]
        
        # Extract mood
        mood = root.find('mood')
        if mood is not None:
            result['mood'] = mood.text
        
        # Extract use cases
        use_cases = root.find('use_cases')
        if use_cases is not None:
            result['use_cases'] = [case.text for case in use_cases.findall('use_case') if case.text]
        
        # Extract suggested filename
        filename = root.find('suggested_filename')
        if filename is not None:
            result['suggested_filename'] = filename.text
        
        return result
        
    except etree.ParseError as e:
        logging.error(f"Error parsing XML response: {str(e)}")
        raise ValueError(f"Invalid XML response: {str(e)}")

def format_xml_output(results: List[Dict[str, Any]], pretty_print: bool = True) -> str:
    """Format results as XML string."""
    try:
        # Create root element
        root = etree.Element("image_analysis_results")
        
        # Add each result as an image element
        for result in results:
            image = etree.SubElement(root, "image")
            
            # Add path information
            if any(key in result for key in ['original_path', 'original_filename', 'new_filename', 'modified_path']):
                path_info = etree.SubElement(image, "path_info")
                for field in ['original_path', 'original_filename', 'new_filename', 'modified_path']:
                    if result.get(field):
                        elem = etree.SubElement(path_info, field)
                        elem.text = str(result[field])
            
            # Add status
            status = etree.SubElement(image, "status")
            success = etree.SubElement(status, "success")
            success.text = str(result.get('success', False)).lower()
            if not result.get('success', False):
                error = etree.SubElement(status, "error")
                error.text = result.get('error', '')
                continue
            
            # Add description
            if result.get('description'):
                description = etree.SubElement(image, "description")
                description.text = result['description']
            
            # Add keywords
            if result.get('keywords'):
                keywords = etree.SubElement(image, "keywords")
                for kw in result['keywords']:
                    if kw:
                        keyword = etree.SubElement(keywords, "keyword")
                        keyword.text = str(kw)
            
            # Add technical details
            if result.get('technical_details'):
                tech_details = etree.SubElement(image, "technical_details")
                if isinstance(result['technical_details'], dict):
                    for key, value in result['technical_details'].items():
                        if value:
                            detail = etree.SubElement(tech_details, key.lower().replace(' ', '_'))
                            detail.text = str(value)
                else:
                    tech_details.text = str(result['technical_details'])
            
            # Add visual elements
            if result.get('visual_elements'):
                visual = etree.SubElement(image, "visual_elements")
                if isinstance(result['visual_elements'], list):
                    for elem in result['visual_elements']:
                        if elem:
                            element = etree.SubElement(visual, "element")
                            element.text = str(elem)
                else:
                    visual.text = str(result['visual_elements'])
            
            # Add composition
            if result.get('composition'):
                composition = etree.SubElement(image, "composition")
                if isinstance(result['composition'], list):
                    for comp in result['composition']:
                        if comp:
                            technique = etree.SubElement(composition, "technique")
                            technique.text = str(comp)
                else:
                    composition.text = str(result['composition'])
            
            # Add mood
            if result.get('mood'):
                mood = etree.SubElement(image, "mood")
                mood.text = str(result['mood'])
            
            # Add use cases
            if result.get('use_cases'):
                use_cases = etree.SubElement(image, "use_cases")
                if isinstance(result['use_cases'], list):
                    for use_case in result['use_cases']:
                        if use_case:
                            case = etree.SubElement(use_cases, "use_case")
                            case.text = str(use_case)
                else:
                    use_cases.text = str(result['use_cases'])
            
            # Add suggested filename
            if result.get('suggested_filename'):
                filename = etree.SubElement(image, "suggested_filename")
                filename.text = str(result['suggested_filename'])
        
        # Convert to string with proper formatting
        return etree.tostring(root, encoding='unicode', pretty_print=pretty_print)
        
    except Exception as e:
        logging.error(f"Error formatting XML output: {str(e)}")
        raise ValueError(f"Error formatting XML output: {str(e)}")