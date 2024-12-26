"""
Image utilities for loading and validating images.
"""
import os
import tempfile
from typing import Optional, Tuple, List
from PIL import Image
import logging
import shutil
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp'}

class ImageValidationError(Exception):
    """Custom exception for image validation errors."""
    pass

class ImageCompressionError(Exception):
    """Custom exception for image compression errors."""
    pass

def is_supported_format(file_path: str) -> bool:
    """
    Check if the file has a supported image format extension.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        bool: True if supported, False otherwise
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_FORMATS

def validate_image(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate an image file by checking its format and attempting to open it.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
        
    if not is_supported_format(file_path):
        return False, f"Unsupported format: {file_path}"
        
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True, None
    except Exception as e:
        return False, f"Invalid or corrupted image: {str(e)}"

def load_image(file_path: str) -> Image.Image:
    """
    Load an image file after validation.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Image.Image: Loaded PIL Image object
        
    Raises:
        ImageValidationError: If image validation fails
    """
    is_valid, error = validate_image(file_path)
    if not is_valid:
        raise ImageValidationError(error)
        
    try:
        return Image.open(file_path)
    except Exception as e:
        raise ImageValidationError(f"Failed to load image: {str(e)}")

def get_image_info(file_path: str) -> dict:
    """
    Get basic information about an image file.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        dict: Dictionary containing image information
        
    Raises:
        ImageValidationError: If image validation fails
    """
    with load_image(file_path) as img:
        return {
            'format': img.format,
            'mode': img.mode,
            'size': img.size,
            'width': img.width,
            'height': img.height,
            'file_size': os.path.getsize(file_path)
        }

def process_image_batch(file_paths: List[str]) -> List[Tuple[str, bool, Optional[str]]]:
    """
    Process a batch of images, validating each one.
    
    Args:
        file_paths: List of paths to image files
        
    Returns:
        List[Tuple[str, bool, Optional[str]]]: List of (file_path, is_valid, error_message)
    """
    results = []
    for file_path in file_paths:
        is_valid, error = validate_image(file_path)
        results.append((file_path, is_valid, error))
        if is_valid:
            logger.info(f"Validated image: {file_path}")
        else:
            logger.warning(f"Invalid image {file_path}: {error}")
    return results

def compress_image(input_path: str, output_path: str, max_dimension: int = 1920, quality: int = 85) -> str:
    """Compress an image while maintaining aspect ratio."""
    try:
        # Start timing
        start_time = time.time()
        
        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Validate quality parameter
        if not 0 <= quality <= 100:
            raise ValueError(f"Quality must be between 0 and 100, got {quality}")

        # Get original file size
        original_size = os.path.getsize(input_path) / (1024 * 1024)  # Convert to MB
        logger.info(f"Starting compression of {os.path.basename(input_path)} ({original_size:.2f} MB)")

        # Open and process image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                logger.info("Converting image to RGB mode")
                img = img.convert('RGB')
                
            # Calculate new dimensions
            width, height = img.size
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save compressed image
            logger.info(f"Saving compressed image with quality {quality}")
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
            
        # Calculate compression stats
        end_time = time.time()
        compressed_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
        compression_ratio = (1 - compressed_size/original_size) * 100
        processing_time = end_time - start_time
        
        logger.info(f"Compression complete:")
        logger.info(f"- Original size: {original_size:.2f} MB")
        logger.info(f"- Compressed size: {compressed_size:.2f} MB")
        logger.info(f"- Compression ratio: {compression_ratio:.1f}%")
        logger.info(f"- Processing time: {processing_time:.2f} seconds")
            
        return output_path
        
    except Exception as e:
        logger.error(f"Error compressing image: {str(e)}")
        raise

def compress_image_batch(
    file_paths: List[str],
    output_dir: Optional[str] = None,
    max_dimension: Optional[int] = None,
    quality: int = 85
) -> List[Tuple[str, bool, Optional[str]]]:
    """
    Compress a batch of images.

    Args:
        file_paths: List of paths to image files
        output_dir: Directory to save compressed images (if None, overwrites originals)
        max_dimension: Maximum width or height (maintains aspect ratio)
        quality: JPEG quality (1-100)

    Returns:
        List[Tuple[str, bool, str]]: List of (file_path, success, error_message/output_path)
    """
    results = []
    for file_path in file_paths:
        try:
            output_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.basename(file_path)
                base, _ = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{base}_compressed.jpg")

            compressed_path = compress_image(
                file_path,
                output_path,
                max_dimension=max_dimension,
                quality=quality
            )
            results.append((file_path, True, compressed_path))
        except Exception as e:
            logger.error(f"Error compressing {file_path}: {str(e)}")
            results.append((file_path, False, str(e)))
    return results

def create_llm_optimized_copy(
    file_path: str,
    max_dimension: int = 1024,
    quality: int = 85,
) -> Tuple[str, str]:
    """
    Creates a temporary compressed copy of an image optimized for LLM processing.

    Args:
        file_path: Path to the input image file
        max_dimension: Maximum width or height (maintains aspect ratio)
        quality: JPEG quality (1-100)

    Returns:
        Tuple[str, str]: (temp_dir_path, compressed_image_path)

    Raises:
        ImageCompressionError: If compression fails
    """
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix='llm_image_')

        # Create compressed copy
        filename = os.path.basename(file_path)
        base, _ = os.path.splitext(filename)
        compressed_path = os.path.join(temp_dir, f"{base}_compressed.jpg")

        compressed_path = compress_image(
            file_path,
            compressed_path,
            max_dimension=max_dimension,
            quality=quality
        )

        return temp_dir, compressed_path

    except Exception as e:
        # Clean up on error
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise ImageCompressionError(f"Failed to create LLM-optimized copy: {str(e)}")

def create_llm_optimized_batch(
    file_paths: List[str],
    max_dimension: int = 1024,
    quality: int = 85,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Creates temporary compressed copies of multiple images optimized for LLM processing.
    
    Args:
        file_paths: List of paths to image files
        max_dimension: Maximum width or height (maintains aspect ratio)
        quality: JPEG quality (1-100)
        
    Returns:
        Tuple[str, List[Tuple[str, str]]]: (temp_dir, list of (original_path, compressed_path))
        
    Raises:
        ImageCompressionError: If compression fails
    """
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix='llm_batch_')
        results = []
        errors = []

        # Process each image
        for file_path in file_paths:
            try:
                # Create compressed copy
                filename = os.path.basename(file_path)
                base, _ = os.path.splitext(filename)
                compressed_path = os.path.join(temp_dir, f"{base}_compressed.jpg")
                
                compressed_path = compress_image(
                    file_path,
                    compressed_path,
                    max_dimension=max_dimension,
                    quality=quality
                )
                
                results.append((file_path, compressed_path))
            except Exception as e:
                logger.error(f"Failed to compress {file_path}: {str(e)}")
                errors.append(str(e))

        if errors:
            # Clean up on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ImageCompressionError(f"Failed to compress some images: {'; '.join(errors)}")

        return temp_dir, results

    except Exception as e:
        # Clean up on error
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise ImageCompressionError(f"Failed to process batch: {str(e)}")

def validate_image_path(path: str) -> Optional[str]:
    """
    Validate that the given path points to a valid image file.
    
    Args:
        path: Path to the image file
        
    Returns:
        Validated path if valid, None otherwise
    """
    if not os.path.exists(path):
        return None
        
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    
    if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
        return None
        
    return path 