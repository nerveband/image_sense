"""
Metadata handler for managing EXIF data using pyexiftool.
"""
import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import exiftool
from ..core.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataError(Exception):
    """Custom exception for metadata handling errors"""
    pass

class MetadataHandler:
    """Handler for reading and writing image metadata."""

    def __init__(self, exiftool_path: Optional[str] = None):
        """Initialize the metadata handler."""
        self.exiftool_path = exiftool_path
        self.et = exiftool.ExifToolHelper(executable=exiftool_path)

    def read_metadata(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Read metadata from an image file."""
        try:
            # Convert path to string
            image_path = str(Path(image_path).resolve())
            metadata = self.et.get_metadata(image_path)
            
            if metadata:
                return {
                    'success': True,
                    'metadata': metadata[0] if isinstance(metadata, list) else metadata,
                    'original_path': image_path,
                    'original_filename': os.path.basename(image_path)
                }
            else:
                return {
                    'success': False,
                    'error': 'No metadata found',
                    'original_path': image_path,
                    'original_filename': os.path.basename(image_path)
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'original_path': str(image_path),
                'original_filename': os.path.basename(str(image_path))
            }

    def write_metadata(self, image_path: Union[str, Path], metadata: Dict[str, Any], backup: bool = True, duplicate: bool = False) -> Dict[str, Any]:
        """Write metadata to an image file."""
        try:
            # Convert path to string
            image_path = str(Path(image_path).resolve())
            
            # Prepare metadata tags
            tags = {}
            
            # Map structured fields to EXIF/IPTC/XMP tags
            if 'description' in metadata:
                tags['-IPTC:Caption-Abstract'] = metadata['description']
                tags['-XMP:Description'] = metadata['description']
                tags['-EXIF:ImageDescription'] = metadata['description']
            
            if 'keywords' in metadata:
                if isinstance(metadata['keywords'], list):
                    tags['-IPTC:Keywords'] = metadata['keywords']
                    tags['-XMP:Subject'] = metadata['keywords']
                else:
                    tags['-IPTC:Keywords'] = [str(metadata['keywords'])]
                    tags['-XMP:Subject'] = [str(metadata['keywords'])]
            
            if 'visual_elements' in metadata:
                if isinstance(metadata['visual_elements'], list):
                    tags['-IPTC:Subject'] = metadata['visual_elements']
                else:
                    tags['-IPTC:Subject'] = [str(metadata['visual_elements'])]
            
            if 'composition' in metadata:
                if isinstance(metadata['composition'], list):
                    tags['-IPTC:Notes'] = '\n'.join(str(c) for c in metadata['composition'])
                else:
                    tags['-IPTC:Notes'] = str(metadata['composition'])
            
            if 'mood' in metadata:
                tags['-IPTC:Category'] = str(metadata['mood'])
                tags['-XMP:Mood'] = str(metadata['mood'])
            
            if 'use_cases' in metadata:
                if isinstance(metadata['use_cases'], list):
                    tags['-XMP:Usage'] = metadata['use_cases']
                else:
                    tags['-XMP:Usage'] = [str(metadata['use_cases'])]
            
            # Add technical details if present
            if 'technical_details' in metadata and isinstance(metadata['technical_details'], dict):
                tech = metadata['technical_details']
                if 'format' in tech:
                    tags['-File:FileType'] = str(tech['format'])
                if 'dimensions' in tech:
                    tags['-File:ImageSize'] = str(tech['dimensions'])
                if 'color_space' in tech:
                    tags['-File:ColorSpace'] = str(tech['color_space'])
            
            # Add software tag
            tags['-XMP:Software'] = 'Image Sense AI Processor'
            
            # Write metadata using exiftool
            try:
                # Prepare command parameters
                params = []
                if not backup:
                    params.append('-overwrite_original')
                
                # Convert tags to exiftool format
                for tag, value in tags.items():
                    if isinstance(value, list):
                        # Handle list values
                        for item in value:
                            if item:  # Only add non-empty values
                                params.append(f"{tag}={str(item)}")
                    else:
                        # Handle single values
                        if value:  # Only add non-empty values
                            params.append(f"{tag}={str(value)}")
                
                if not params:
                    return {
                        'success': False,
                        'error': 'No valid metadata tags to write',
                        'original_path': image_path,
                        'original_filename': os.path.basename(image_path)
                    }
                
                # Execute exiftool command
                self.et.execute(*params, image_path)
                
                # Verify the metadata was written
                verify_result = self.read_metadata(image_path)
                
                if verify_result.get('success', False):
                    return {
                        'success': True,
                        'original_path': image_path,
                        'original_filename': os.path.basename(image_path),
                        'modified_path': image_path
                    }
                
                return {
                    'success': False,
                    'error': 'Failed to verify metadata was written',
                    'original_path': image_path,
                    'original_filename': os.path.basename(image_path)
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Failed to write metadata: {str(e)}',
                    'original_path': image_path,
                    'original_filename': os.path.basename(image_path)
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Error in write_metadata: {str(e)}',
                'original_path': str(image_path),
                'original_filename': os.path.basename(str(image_path))
            }

    def copy_metadata(self, source_path: Union[str, Path], target_path: Union[str, Path], 
                     tags: Optional[List[str]] = None) -> None:
        """
        Copy metadata from one image to another.
        
        Args:
            source_path: Path to the source image
            target_path: Path to the target image
            tags: Optional list of specific tags to copy
        """
        try:
            source_path = str(Path(source_path).resolve())
            target_path = str(Path(target_path).resolve())

            params = ['-overwrite_original', '-TagsFromFile', source_path]
            if tags:
                # Map internal tag names to ExifTool tag names
                exif_tags = [self.TAG_MAP.get(tag, tag) for tag in tags]
                params.extend(exif_tags)

            self.et.execute(*params, target_path)

        except Exception as e:
            raise MetadataError(f"Failed to copy metadata: {str(e)}")

    def remove_metadata(self, image_path: Union[str, Path], tags: Optional[List[str]] = None, 
                       backup: bool = True) -> None:
        """
        Remove metadata from an image file.
        
        Args:
            image_path: Path to the image file
            tags: Optional list of specific tags to remove
            backup: Whether to create a backup of the original file
        """
        try:
            image_path = str(Path(image_path).resolve())
            
            params = []
            if not backup:
                params.append('-overwrite_original')

            if tags:
                # Map internal tag names to ExifTool tag names and add = to clear them
                exif_tags = {self.TAG_MAP.get(tag, tag).lstrip('-'): '' for tag in tags}
                self.et.set_tags(image_path, exif_tags, params=params)
            else:
                # Remove all metadata
                self.et.execute('-all=', '-overwrite_original' if not backup else '', image_path)

        except Exception as e:
            raise MetadataError(f"Failed to remove metadata: {str(e)}")

    def process_batch(self, image_paths: List[Union[str, Path]], operation: str,
                     metadata: Optional[Dict[str, Any]] = None,
                     tags: Optional[List[str]] = None,
                     output_format: str = 'csv',
                     output_path: Optional[str] = None,
                     backup: bool = True,
                     duplicate: bool = False) -> List[Dict[str, Any]]:
        """
        Process a batch of images with the same operation.
        
        Args:
            image_paths: List of paths to image files
            operation: Operation to perform ('read', 'write', 'remove')
            metadata: Metadata to write (for 'write' operation)
            tags: Optional list of specific tags
            output_format: Format for saving results ('csv' or 'xml')
            output_path: Path to save results
            backup: Whether to create backups
            duplicate: Whether to create duplicates before modifying
            
        Returns:
            List of results
        """
        results = []
        
        # Convert paths to absolute paths and store original paths
        processed_paths = []
        for path in image_paths:
            path = Path(path).resolve()
            processed_paths.append({
                'original_path': str(path),
                'original_filename': path.name,
                'new_filename': None,
                'modified_path': None,
                'success': False
            })

        try:
            if operation == 'read':
                # Read metadata from all files at once
                batch_results = self.et.get_metadata([str(p) for p in processed_paths])
                for i, metadata_result in enumerate(batch_results):
                    result = {
                        **processed_paths[i],
                        'success': True,
                        'metadata': metadata_result
                    }
                    
                    # Extract metadata fields
                    result.update(self._extract_metadata_fields(metadata_result))
                    results.append(result)

            elif operation == 'write' and metadata:
                # Process each file individually for write operations
                for i, path_info in enumerate(processed_paths):
                    result = self.write_metadata(
                        path_info['original_path'],
                        metadata,
                        backup,
                        duplicate
                    )
                    # Merge with original path info and metadata
                    result.update(path_info)
                    if metadata:
                        result.update(self._extract_metadata_fields(metadata))
                    results.append(result)

            elif operation == 'remove':
                # Remove metadata from all files
                for i, path_info in enumerate(processed_paths):
                    try:
                        self.remove_metadata(path_info['original_path'], tags, backup)
                        results.append({
                            **path_info,
                            'success': True
                        })
                    except Exception as e:
                        results.append({
                            **path_info,
                            'success': False,
                            'error': str(e)
                        })

            # Save results if output path provided
            if output_path:
                if output_format == 'csv':
                    self._save_to_csv(results, output_path)
                elif output_format == 'xml':
                    self._save_to_xml(results, output_path)

            return results

        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            # Return results collected so far with error status
            for result in results:
                result['success'] = False
                result['error'] = str(e)
            return results

    def _extract_metadata_fields(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract standardized metadata fields from raw metadata.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Dict containing extracted and standardized metadata fields
        """
        result = {
            'description': '',
            'keywords': [],
            'visual_elements': [],
            'composition': [],
            'mood': '',
            'use_cases': [],
            'technical_details': {}
        }
        
        # Clean markdown formatting
        def clean_text(text: str) -> str:
            return text.replace('*', '').replace('`', '').replace('**', '').strip()
        
        # Extract description from various possible tags
        for desc_tag in ['IPTC:Caption-Abstract', 'XMP:Description', 'EXIF:ImageDescription']:
            if desc_tag in metadata:
                result['description'] = clean_text(str(metadata[desc_tag]))
                break
        
        # Extract keywords from various possible tags
        for kw_tag in ['IPTC:Keywords', 'XMP:Subject']:
            if kw_tag in metadata:
                keywords = metadata[kw_tag]
                if isinstance(keywords, list):
                    result['keywords'].extend([clean_text(k) for k in keywords if k])
                else:
                    result['keywords'].append(clean_text(str(keywords)))
                break
        
        # Extract visual elements from Subject tag
        if 'IPTC:Subject' in metadata:
            elements = metadata['IPTC:Subject']
            if isinstance(elements, list):
                result['visual_elements'].extend([clean_text(e) for e in elements if e])
            else:
                result['visual_elements'].append(clean_text(str(elements)))
        
        # Extract composition from Notes tag
        if 'IPTC:Notes' in metadata:
            notes = metadata['IPTC:Notes']
            if isinstance(notes, str):
                # Split on newlines and clean each line
                result['composition'] = [clean_text(line) for line in notes.split('\n') if line.strip()]
            elif isinstance(notes, list):
                result['composition'].extend([clean_text(n) for n in notes if n])
            else:
                result['composition'].append(clean_text(str(notes)))
        
        # Extract mood from various possible tags
        for mood_tag in ['IPTC:Category', 'XMP:Mood']:
            if mood_tag in metadata:
                result['mood'] = clean_text(str(metadata[mood_tag]))
                break
        
        # Extract use cases from Usage tag
        if 'XMP:Usage' in metadata:
            cases = metadata['XMP:Usage']
            if isinstance(cases, list):
                result['use_cases'].extend([clean_text(c) for c in cases if c])
            else:
                result['use_cases'].append(clean_text(str(cases)))
        
        # Extract technical details
        tech_fields = {
            'format': 'File:FileType',
            'dimensions': 'File:ImageSize',
            'color_space': 'File:ColorSpace'
        }
        for key, tag in tech_fields.items():
            if tag in metadata:
                result['technical_details'][key] = clean_text(str(metadata[tag]))
        
        return result

    def _save_to_csv(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """Save results to a CSV file."""
        import csv
        
        # Define CSV headers based on expected fields
        headers = [
            'original_path', 'original_filename', 'new_filename', 'modified_path',
            'suggested_filename', 'success', 'error', 'description', 'keywords',
            'visual_elements', 'composition', 'mood', 'use_cases',
            'format', 'dimensions', 'color_space'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            
            for result in results:
                # Clean markdown formatting
                def clean_text(text: str) -> str:
                    return text.replace('*', '').replace('`', '').replace('**', '').strip()
                
                # Prepare row data
                row = {
                    'original_path': result.get('original_path', ''),
                    'original_filename': result.get('original_filename', ''),
                    'new_filename': result.get('new_filename', ''),
                    'modified_path': result.get('modified_path', ''),
                    'suggested_filename': clean_text(result.get('suggested_filename', '')),
                    'success': result.get('success', False),
                    'error': result.get('error', ''),
                    'description': clean_text(result.get('description', ''))
                }
                
                # Handle list fields by joining with semicolons
                for field in ['keywords', 'visual_elements', 'composition', 'use_cases']:
                    values = result.get(field, [])
                    if isinstance(values, (list, tuple)):
                        row[field] = '; '.join(clean_text(str(v)) for v in values if v)
                    else:
                        row[field] = clean_text(str(values))
                
                # Handle mood field
                row['mood'] = clean_text(result.get('mood', ''))
                
                # Handle technical details
                tech = result.get('technical_details', {})
                row.update({
                    'format': clean_text(tech.get('format', '')),
                    'dimensions': clean_text(tech.get('dimensions', '')),
                    'color_space': clean_text(tech.get('color_space', ''))
                })
                
                writer.writerow(row)
                
        logger.info(f"Results saved to CSV: {output_path}")

    def _save_to_xml(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to XML file."""
        from lxml import etree
        
        # Create root element
        root = etree.Element("image_analysis_results")
        
        for result in results:
            image = etree.SubElement(root, "image")
            
            # Add file information
            file_info = etree.SubElement(image, "file_info")
            for field in ['original_path', 'original_filename', 'new_filename', 'modified_path', 'suggested_filename']:
                if result.get(field):
                    elem = etree.SubElement(file_info, field)
                    elem.text = str(result[field])
            
            # Add success and error info
            status = etree.SubElement(image, "status")
            success = etree.SubElement(status, "success")
            success.text = str(result.get('success', False))
            
            if 'error' in result:
                error = etree.SubElement(status, "error")
                error.text = str(result['error'])
            
            # Add description
            if 'description' in result:
                desc = etree.SubElement(image, "description")
                desc.text = str(result['description'])
            
            # Add keywords
            if 'keywords' in result:
                keywords = etree.SubElement(image, "keywords")
                for keyword in result['keywords']:
                    if keyword:
                        kw = etree.SubElement(keywords, "keyword")
                        kw.text = str(keyword).strip()
            
            # Add technical details
            if 'technical_details' in result:
                tech = etree.SubElement(image, "technical_details")
                for key, value in result['technical_details'].items():
                    if value:  # Only add non-empty values
                        detail = etree.SubElement(tech, key)
                        detail.text = str(value)
            
            # Add visual elements
            if 'visual_elements' in result:
                elements = etree.SubElement(image, "visual_elements")
                for element in result['visual_elements']:
                    if element:
                        elem = etree.SubElement(elements, "element")
                        elem.text = str(element).strip()
            
            # Add composition
            if 'composition' in result:
                comp = etree.SubElement(image, "composition")
                for technique in result['composition']:
                    if technique:
                        tech = etree.SubElement(comp, "technique")
                        tech.text = str(technique).strip()
            
            # Add mood
            if result.get('mood'):
                mood = etree.SubElement(image, "mood")
                mood.text = str(result['mood'])
            
            # Add use cases
            if 'use_cases' in result:
                uses = etree.SubElement(image, "use_cases")
                for use_case in result['use_cases']:
                    if use_case:
                        case = etree.SubElement(uses, "use_case")
                        case.text = str(use_case).strip()
            
            # Add metadata if present
            if 'metadata' in result:
                metadata = etree.SubElement(image, "metadata")
                for key, value in result['metadata'].items():
                    if value:  # Only add non-empty values
                        tag = etree.SubElement(metadata, key.replace(':', '_'))
                        if isinstance(value, (list, tuple)):
                            tag.text = '; '.join(str(v) for v in value if v)
                        else:
                            tag.text = str(value)
        
        # Save to file with pretty formatting
        tree = etree.ElementTree(root)
        tree.write(output_path, pretty_print=True, xml_declaration=True, encoding='utf-8')
        logger.info(f"Results saved to XML: {output_path}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if hasattr(self, 'et'):
            self.et.terminate() 