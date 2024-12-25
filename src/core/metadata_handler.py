"""
Metadata handler for managing EXIF data using exiftool.
"""
import os
import sys
import json
import subprocess
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataError(Exception):
    """Custom exception for metadata-related errors."""
    pass

class ExifTool:
    """
    A class to handle interaction with ExifTool for reading and writing image metadata.
    """
    def __init__(self, exiftool_path: Optional[str] = None):
        """
        Initialize ExifTool handler.
        
        Args:
            exiftool_path: Optional path to exiftool executable
        """
        self.exiftool_path = self._find_exiftool(exiftool_path)
        self._verify_exiftool()

    def _find_exiftool(self, custom_path: Optional[str] = None) -> str:
        """
        Find the exiftool executable.
        
        Args:
            custom_path: Optional custom path to exiftool
            
        Returns:
            str: Path to exiftool executable
        """
        if custom_path and os.path.isfile(custom_path):
            return custom_path

        # Check system PATH first
        if sys.platform == 'win32':
            exiftool_name = 'exiftool.exe'
        else:
            exiftool_name = 'exiftool'

        for path in os.environ.get('PATH', '').split(os.pathsep):
            exe_path = os.path.join(path, exiftool_name)
            if os.path.isfile(exe_path):
                return exe_path

        # Fall back to bundled exiftool if system one not found
        bundled_path = self._get_bundled_exiftool_path()
        if bundled_path:
            return bundled_path

        raise MetadataError("ExifTool not found. Please ensure it's installed or bundled correctly.")

    def _get_bundled_exiftool_path(self) -> Optional[str]:
        """
        Get the path to the bundled exiftool executable.
        """
        # Determine platform-specific executable name
        if sys.platform == 'win32':
            exe_name = 'exiftool.exe'
        elif sys.platform == 'darwin':
            exe_name = 'exiftool'
        else:  # Linux/Unix
            exe_name = 'exiftool'

        # Check relative to the application root
        bundled_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'resources',
            'exiftool',
            exe_name
        )

        if os.path.isfile(bundled_path):
            # Ensure the bundled executable has correct permissions
            if sys.platform != 'win32':
                try:
                    os.chmod(bundled_path, 0o755)
                except OSError as e:
                    logger.warning(f"Failed to set executable permissions: {e}")
            return bundled_path

        return None

    def _verify_exiftool(self):
        """
        Verify that exiftool is working correctly.
        """
        try:
            result = subprocess.run(
                [self.exiftool_path, '-ver'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"ExifTool version {result.stdout.strip()} found at {self.exiftool_path}")
        except subprocess.CalledProcessError as e:
            raise MetadataError(f"Failed to verify ExifTool: {str(e)}")

    def read_metadata(self, image_path: Union[str, Path], 
                     tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Read metadata from an image file.
        
        Args:
            image_path: Path to the image file
            tags: Optional list of specific tags to read
            
        Returns:
            Dict containing the metadata
        """
        cmd = [self.exiftool_path, '-j']
        if tags:
            cmd.extend([f'-{tag}' for tag in tags])
        cmd.append(str(image_path))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            metadata = json.loads(result.stdout)[0]  # ExifTool returns a list with one item
            return metadata
        except subprocess.CalledProcessError as e:
            raise MetadataError(f"Failed to read metadata: {str(e)}")
        except json.JSONDecodeError as e:
            raise MetadataError(f"Failed to parse metadata: {str(e)}")

    def write_metadata(self, image_path: Union[str, Path], 
                      metadata: Dict[str, Any],
                      backup: bool = True) -> None:
        """
        Write metadata to an image file.
        
        Args:
            image_path: Path to the image file
            metadata: Dictionary of metadata to write
            backup: Whether to create a backup of the original file
        """
        cmd = [self.exiftool_path]
        if not backup:
            cmd.append('-overwrite_original')

        # Convert metadata dict to ExifTool arguments
        for tag, value in metadata.items():
            if value is not None:
                cmd.extend([f'-{tag}={value}'])

        cmd.append(str(image_path))

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise MetadataError(f"Failed to write metadata: {str(e)}")

    def copy_metadata(self, source_path: Union[str, Path],
                     target_path: Union[str, Path],
                     tags: Optional[List[str]] = None) -> None:
        """
        Copy metadata from one image to another.
        
        Args:
            source_path: Path to the source image
            target_path: Path to the target image
            tags: Optional list of specific tags to copy
        """
        cmd = [self.exiftool_path, '-overwrite_original']
        if tags:
            cmd.extend([f'-{tag}' for tag in tags])
        cmd.extend(['-TagsFromFile', str(source_path), str(target_path)])

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise MetadataError(f"Failed to copy metadata: {str(e)}")

    def remove_metadata(self, image_path: Union[str, Path],
                       tags: Optional[List[str]] = None,
                       backup: bool = True) -> None:
        """
        Remove metadata from an image file.
        
        Args:
            image_path: Path to the image file
            tags: Optional list of specific tags to remove (removes all if None)
            backup: Whether to create a backup of the original file
        """
        cmd = [self.exiftool_path]
        if not backup:
            cmd.append('-overwrite_original')

        if tags:
            cmd.extend([f'-{tag}=' for tag in tags])
        else:
            cmd.append('-all=')  # Remove all metadata

        cmd.append(str(image_path))

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise MetadataError(f"Failed to remove metadata: {str(e)}")

    def process_batch(self, image_paths: List[Union[str, Path]], 
                     operation: str,
                     metadata: Optional[Dict[str, Any]] = None,
                     tags: Optional[List[str]] = None,
                     output_format: str = 'csv',
                     output_path: Optional[str] = None,
                     backup: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of images with the same operation and save structured output.
        
        Args:
            image_paths: List of paths to image files
            operation: Operation to perform ('read', 'write', 'remove')
            metadata: List of metadata dictionaries or single metadata dict
            tags: Optional list of specific tags
            output_format: Format to save results ('csv' or 'xml')
            output_path: Path to save output file
            backup: Whether to create backups (for 'write' and 'remove' operations)
            
        Returns:
            List of results (for 'read' operation) or empty list (for other operations)
        """
        results = []
        
        # Handle metadata as list or single dict
        if isinstance(metadata, list):
            metadata_list = metadata
        else:
            metadata_list = [metadata] * len(image_paths) if metadata else []
        
        for i, path in enumerate(image_paths):
            try:
                if operation == 'read':
                    result = self.read_metadata(path, tags)
                    results.append(result)
                elif operation == 'write' and metadata_list:
                    # Get metadata for this image
                    img_metadata = metadata_list[i]
                    if img_metadata.get('success', True):  # Only write if successful
                        # Convert structured data to flat metadata
                        exif_metadata = {
                            'Description': img_metadata.get('description', ''),
                            'Keywords': ', '.join(img_metadata.get('keywords', [])),
                            'Technical': str(img_metadata.get('technical_details', {})),
                            'VisualElements': ', '.join(img_metadata.get('visual_elements', [])),
                            'Composition': ', '.join(img_metadata.get('composition', [])),
                            'Mood': img_metadata.get('mood', ''),
                            'UseCases': ', '.join(img_metadata.get('use_cases', [])),
                            'Software': 'Image Sense AI Processor'
                        }
                        self.write_metadata(path, exif_metadata, backup)
                elif operation == 'remove':
                    self.remove_metadata(path, tags, backup)
                logger.info(f"Successfully processed {path}")
            except Exception as e:
                logger.error(f"Failed to process {path}: {str(e)}")
                results.append({'error': str(e), 'path': str(path)})

        # Save structured output if path provided
        if output_path and metadata_list:
            if output_format == 'csv':
                self._save_to_csv(metadata_list, output_path)
            elif output_format == 'xml':
                self._save_to_xml(metadata_list, output_path)

        return results

    def _save_to_csv(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to CSV file."""
        import pandas as pd
        
        # Flatten nested structures for CSV
        flattened_results = []
        for result in results:
            flat_result = {
                'path': result.get('path', ''),
                'success': result.get('success', False),
                'description': result.get('description', ''),
                'keywords': ','.join(result.get('keywords', [])),
            }
            
            # Add technical details if present
            if 'technical_details' in result:
                for key, value in result['technical_details'].items():
                    flat_result[f'technical_{key}'] = value
                    
            # Add other list fields
            for field in ['visual_elements', 'composition', 'use_cases']:
                if field in result:
                    flat_result[field] = ','.join(result[field])
                    
            # Add mood if present
            if 'mood' in result:
                flat_result['mood'] = result['mood']
                
            flattened_results.append(flat_result)
            
        # Convert to DataFrame and save
        df = pd.DataFrame(flattened_results)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to CSV: {output_path}")

    def _save_to_xml(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to XML file."""
        from lxml import etree
        
        # Create root element
        root = etree.Element("image_analysis_results")
        
        for result in results:
            image = etree.SubElement(root, "image")
            
            # Add basic fields
            path = etree.SubElement(image, "path")
            path.text = str(result.get('path', ''))
            
            success = etree.SubElement(image, "success")
            success.text = str(result.get('success', False))
            
            if 'description' in result:
                desc = etree.SubElement(image, "description")
                desc.text = result['description']
            
            # Add keywords
            if 'keywords' in result:
                keywords = etree.SubElement(image, "keywords")
                for kw in result['keywords']:
                    keyword = etree.SubElement(keywords, "keyword")
                    keyword.text = kw
            
            # Add technical details
            if 'technical_details' in result:
                tech = etree.SubElement(image, "technical_details")
                for key, value in result['technical_details'].items():
                    detail = etree.SubElement(tech, key)
                    detail.text = str(value)
            
            # Add list fields
            for field in ['visual_elements', 'composition', 'use_cases']:
                if field in result:
                    elements = etree.SubElement(image, field)
                    for item in result[field]:
                        element = etree.SubElement(elements, "item")
                        element.text = item
            
            # Add mood if present
            if 'mood' in result:
                mood = etree.SubElement(image, "mood")
                mood.text = result['mood']
        
        # Save to file
        tree = etree.ElementTree(root)
        tree.write(output_path, pretty_print=True, xml_declaration=True, encoding='utf-8')
        logger.info(f"Saved results to XML: {output_path}")

class MetadataHandler:
    """
    High-level interface for metadata operations.
    Wraps the ExifTool class with additional functionality.
    """
    def __init__(self, exiftool_path: Optional[str] = None):
        """
        Initialize the metadata handler.
        
        Args:
            exiftool_path: Optional path to exiftool executable
        """
        self.exiftool = ExifTool(exiftool_path)

    def read_metadata(self, image_path: Union[str, Path], tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Read metadata from an image file.
        
        Args:
            image_path: Path to the image file
            tags: Optional list of specific tags to read
            
        Returns:
            Dict containing the metadata
        """
        return self.exiftool.read_metadata(image_path, tags)

    def write_metadata(self, image_path: Union[str, Path], metadata: Dict[str, Any], backup: bool = True) -> None:
        """
        Write metadata to an image file.
        
        Args:
            image_path: Path to the image file
            metadata: Dictionary of metadata to write
            backup: Whether to create a backup of the original file
        """
        self.exiftool.write_metadata(image_path, metadata, backup)

    def copy_metadata(self, source_path: Union[str, Path], target_path: Union[str, Path], 
                     tags: Optional[List[str]] = None) -> None:
        """
        Copy metadata from one image to another.
        
        Args:
            source_path: Path to the source image
            target_path: Path to the target image
            tags: Optional list of specific tags to copy
        """
        self.exiftool.copy_metadata(source_path, target_path, tags)

    def remove_metadata(self, image_path: Union[str, Path], tags: Optional[List[str]] = None, 
                       backup: bool = True) -> None:
        """
        Remove metadata from an image file.
        
        Args:
            image_path: Path to the image file
            tags: Optional list of specific tags to remove (removes all if None)
            backup: Whether to create a backup of the original file
        """
        self.exiftool.remove_metadata(image_path, tags, backup)

    def process_batch(self, image_paths: List[Union[str, Path]], 
                     operation: str,
                     metadata: Optional[Dict[str, Any]] = None,
                     tags: Optional[List[str]] = None,
                     output_format: str = 'csv',
                     output_path: Optional[str] = None,
                     backup: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of images with the same operation and save structured output.
        
        Args:
            image_paths: List of paths to image files
            operation: Operation to perform ('read', 'write', 'remove')
            metadata: List of metadata dictionaries or single metadata dict
            tags: Optional list of specific tags
            output_format: Format to save results ('csv' or 'xml')
            output_path: Path to save output file
            backup: Whether to create backups (for 'write' and 'remove' operations)
            
        Returns:
            List of results (for 'read' operation) or empty list (for other operations)
        """
        return self.exiftool.process_batch(image_paths, operation, metadata, tags, output_format, output_path, backup) 