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

def compress_image(
    file_path: str,
    output_path: Optional[str] = None,
    max_dimension: Optional[int] = None,
    quality: int = 85,
    optimize: bool = True
) -> str:
    """
    Compress and optionally resize an image while maintaining aspect ratio.
    
    Args:
        file_path: Path to the input image file
        output_path: Path to save the compressed image (if None, overwrites original)
        max_dimension: Maximum width or height (maintains aspect ratio)
        quality: JPEG quality (1-100, higher is better quality but larger size)
        optimize: Whether to optimize the output file size
        
    Returns:
        str: Path to the compressed image
        
    Raises:
        ImageCompressionError: If compression fails
    """
    try:
        with load_image(file_path) as img:
            # Convert to RGB if necessary (for PNG with transparency)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1])
                img = background

            # Resize if max_dimension is specified
            if max_dimension:
                width, height = img.size
                if width > max_dimension or height > max_dimension:
                    if width > height:
                        new_width = max_dimension
                        new_height = int(height * (max_dimension / width))
                    else:
                        new_height = max_dimension
                        new_width = int(width * (max_dimension / height))
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Determine output path
            if not output_path:
                output_path = file_path

            # Save with compression
            img.save(
                output_path,
                'JPEG',
                quality=quality,
                optimize=optimize,
                progressive=True
            )
            
            logger.info(f"Compressed image saved to: {output_path}")
            return output_path

    except Exception as e:
        raise ImageCompressionError(f"Failed to compress image: {str(e)}")

def compress_image_batch(
    file_paths: List[str],
    output_dir: Optional[str] = None,
    max_dimension: Optional[int] = None,
    quality: int = 85,
    optimize: bool = True
) -> List[Tuple[str, bool, Optional[str]]]:
    """
    Compress a batch of images.
    
    Args:
        file_paths: List of paths to image files
        output_dir: Directory to save compressed images (if None, overwrites originals)
        max_dimension: Maximum width or height (maintains aspect ratio)
        quality: JPEG quality (1-100)
        optimize: Whether to optimize the output file size
        
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
                max_dimension,
                quality,
                optimize
            )
            results.append((file_path, True, compressed_path))
            
        except (ImageValidationError, ImageCompressionError) as e:
            results.append((file_path, False, str(e)))
            logger.error(f"Failed to compress {file_path}: {str(e)}")
            
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
        
        compress_image(
            file_path,
            compressed_path,
            max_dimension=max_dimension,
            quality=quality,
            optimize=True
        )
        
        return temp_dir, compressed_path
        
    except Exception as e:
        # Clean up on error
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise ImageCompressionError(f"Failed to create LLM-optimized copy: {str(e)}")

def cleanup_llm_optimized_copy(temp_dir: str):
    """
    Removes the temporary directory and its contents.
    
    Args:
        temp_dir: Path to the temporary directory to remove
    """
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory {temp_dir}: {str(e)}")

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
        
        for file_path in file_paths:
            try:
                filename = os.path.basename(file_path)
                base, _ = os.path.splitext(filename)
                compressed_path = os.path.join(temp_dir, f"{base}_compressed.jpg")
                
                compress_image(
                    file_path,
                    compressed_path,
                    max_dimension=max_dimension,
                    quality=quality,
                    optimize=True
                )
                
                results.append((file_path, compressed_path))
                
            except (ImageValidationError, ImageCompressionError) as e:
                logger.error(f"Failed to compress {file_path}: {str(e)}")
                # Continue with other images even if one fails
                
        return temp_dir, results
        
    except Exception as e:
        # Clean up on error
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise ImageCompressionError(f"Failed to create LLM-optimized batch: {str(e)}") 

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