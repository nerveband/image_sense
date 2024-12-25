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
                     backup: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of images with the same operation.
        
        Args:
            image_paths: List of paths to image files
            operation: Operation to perform ('read', 'write', 'remove')
            metadata: Metadata to write (for 'write' operation)
            tags: Optional list of specific tags
            backup: Whether to create backups (for 'write' and 'remove' operations)
            
        Returns:
            List of results (for 'read' operation) or empty list (for other operations)
        """
        results = []
        for path in image_paths:
            try:
                if operation == 'read':
                    result = self.read_metadata(path, tags)
                    results.append(result)
                elif operation == 'write' and metadata:
                    self.write_metadata(path, metadata, backup)
                elif operation == 'remove':
                    self.remove_metadata(path, tags, backup)
                logger.info(f"Successfully processed {path}")
            except Exception as e:
                logger.error(f"Failed to process {path}: {str(e)}")
                results.append({'error': str(e), 'path': str(path)})

        return results 

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

    def process_batch(self, image_paths: List[Union[str, Path]], operation: str,
                     metadata: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None,
                     backup: bool = True) -> List[Dict[str, Any]]:
        """
        Process a batch of images with the same operation.
        
        Args:
            image_paths: List of paths to image files
            operation: Operation to perform ('read', 'write', 'remove')
            metadata: Metadata to write (for 'write' operation)
            tags: Optional list of specific tags
            backup: Whether to create backups (for 'write' and 'remove' operations)
            
        Returns:
            List of results (for 'read' operation) or empty list (for other operations)
        """
        return self.exiftool.process_batch(image_paths, operation, metadata, tags, backup) 