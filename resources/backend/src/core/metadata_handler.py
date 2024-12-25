"""
Metadata handler for processing image metadata using ExifTool
"""

import os
import json
import subprocess
from typing import Dict, Any, List, Optional

class MetadataHandler:
    """Handles reading and writing image metadata using ExifTool"""
    
    def __init__(self, exiftool_path: Optional[str] = None):
        """
        Initialize the metadata handler
        
        Args:
            exiftool_path: Optional path to ExifTool executable
        """
        self.exiftool_path = exiftool_path or 'exiftool'
    
    def read_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        Read metadata from an image file
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Dictionary containing the image metadata
        """
        try:
            # Run ExifTool to get metadata in JSON format
            result = subprocess.run(
                [self.exiftool_path, '-j', '-n', image_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse JSON output
            metadata = json.loads(result.stdout)[0]
            return {
                'success': True,
                'metadata': metadata
            }
            
        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'error': f'ExifTool error: {e.stderr}'
            }
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': f'Failed to parse metadata: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def write_metadata(self, image_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write metadata to an image file
        
        Args:
            image_path: Path to the image file
            metadata: Dictionary containing metadata to write
        
        Returns:
            Dictionary indicating success or failure
        """
        try:
            # Create temporary JSON file with metadata
            temp_json = f'{image_path}_metadata.json'
            with open(temp_json, 'w') as f:
                json.dump([metadata], f)
            
            # Run ExifTool to write metadata from JSON
            result = subprocess.run(
                [self.exiftool_path, '-json', temp_json, '-overwrite_original', image_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Clean up temporary file
            os.remove(temp_json)
            
            return {
                'success': True,
                'message': 'Metadata written successfully'
            }
            
        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'error': f'ExifTool error: {e.stderr}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_batch(self, image_paths: List[str], metadata: Dict[str, Any],
                     callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Process metadata for a batch of images
        
        Args:
            image_paths: List of paths to image files
            metadata: Dictionary containing metadata to write
            callback: Optional callback function for progress updates
        
        Returns:
            List of dictionaries containing results for each image
        """
        results = []
        total = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            result = self.write_metadata(image_path, metadata)
            results.append(result)
            
            if callback:
                progress = ((i + 1) / total) * 100
                callback(progress, image_path)
        
        return results
    
    def validate_exiftool(self) -> bool:
        """
        Validate that ExifTool is available and working
        
        Returns:
            bool: True if ExifTool is working, False otherwise
        """
        try:
            subprocess.run(
                [self.exiftool_path, '-ver'],
                capture_output=True,
                check=True
            )
            return True
        except Exception:
            return False 