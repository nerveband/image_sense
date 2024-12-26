"""
Image processor module for handling image analysis and metadata operations.
"""
import os
import sys
import json
import time
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import typing_extensions as typing
import google.generativeai as genai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TransferSpeedColumn
from lxml import etree
import shutil
import absl.logging

from .image_utils import compress_image
from .llm_handler import get_provider
from .metadata_handler import MetadataHandler
from .config import Config
from .output_handler import OutputHandler
from .ascii_art import BANNER

# Disable absl logging to stderr
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
# Ensure all loggers show debug output
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

console = Console()

# Define schema for image analysis
class TechnicalDetails(typing.TypedDict):
    format: str
    dimensions: str
    color_space: str

class ImageAnalysis(typing.TypedDict):
    description: str
    keywords: list[str]
    technical_details: TechnicalDetails
    visual_elements: list[str]
    composition: list[str]
    mood: str
    use_cases: list[str]
    suggested_filename: str

class ImageProcessingError(Exception):
    """Exception raised when image processing fails."""
    pass

class ImageProcessor:
    """Image processing class that handles analysis and metadata operations."""

    # Prompt template for image analysis
    PROMPT = """Analyze the provided images and return the information in the following format. Use the exact markers and format shown:

[START]
DESCRIPTION: A clear, concise description of the main subject and scene. Use a single line without line breaks.

KEYWORDS:
- keyword1
- keyword2

TECHNICAL_DETAILS:
FORMAT: JPEG
DIMENSIONS: dimensions in pixels
COLOR_SPACE: color space info

VISUAL_ELEMENTS:
- element1
- element2

COMPOSITION:
- technique1
- technique2

MOOD: overall mood or atmosphere in a single line

USE_CASES:
- use case 1
- use case 2

FILENAME: descriptive_filename_with_underscores.jpg
[END]

Important:
1. Return ONLY the content between [START] and [END] markers
2. Use the exact field names and format shown above
3. Each field should be on a new line
4. List items should start with a hyphen and space
5. Keep all text content on a single line (no line breaks within descriptions)"""

    # Define the schema for structured output
    RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "A clear, concise description of the main subject and scene"
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of relevant keywords"
            },
            "technical_details": {
                "type": "object",
                "properties": {
                    "format": {"type": "string"},
                    "dimensions": {"type": "string"},
                    "color_space": {"type": "string"}
                },
                "required": ["format", "dimensions", "color_space"]
            },
            "visual_elements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of key visual elements"
            },
            "composition": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of composition techniques"
            },
            "mood": {
                "type": "string",
                "description": "Overall mood or atmosphere"
            },
            "use_cases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of potential applications"
            },
            "suggested_filename": {
                "type": "string",
                "description": "Descriptive filename with underscores"
            }
        },
        "required": [
            "description",
            "keywords",
            "technical_details",
            "visual_elements",
            "composition",
            "mood",
            "use_cases",
            "suggested_filename"
        ]
    }

    # Content template for structured output
    CONTENT_TEMPLATE = {
        "text": """Analyze these images and provide a SINGLE valid JSON object following this exact schema. Do not include any text before or after the JSON.

{
    "description": string,  // Single line without breaks
    "keywords": string[],  // Array of strings
    "technical_details": {
        "format": string,  // e.g., "JPEG"
        "dimensions": string,  // e.g., "1920x1080"
        "color_space": string  // e.g., "sRGB"
    },
    "visual_elements": string[],  // Array of strings
    "composition": string[],  // Array of strings
    "mood": string,  // Single line without breaks
    "use_cases": string[],  // Array of strings
    "suggested_filename": string  // With underscores
}

Important:
1. Return ONLY the JSON object
2. Use proper JSON formatting
3. No line breaks in strings
4. Arrays must use []
5. Quote all strings
6. No trailing commas"""
    }

    # Available models based on latest documentation
    AVAILABLE_MODELS = {
        '2-flash': 'gemini-2.0-flash-exp',     # Experimental next-gen features
        '1.5-flash': 'gemini-1.5-flash',       # Fast and versatile, up to 3000 images
        '1.5-flash-8b': 'gemini-1.5-flash-8b', # High volume, lower intelligence
        '1.5-pro': 'gemini-1.5-pro',           # Complex reasoning
        'pro': 'gemini-1.5-pro',               # Alias for 1.5-pro
    }

    def __init__(self, api_key: str, config: Optional[Config] = None, model: Optional[str] = None, 
                 rename_files: bool = False, prefix: Optional[str] = None, batch_size: Optional[int] = None,
                 progress_callback: Optional[callable] = None):
        """Initialize the image processor.
        
        Args:
            api_key: API key for the service
            config: Optional configuration object
            model: Optional model name to use
            rename_files: Whether to rename processed files
            prefix: Optional prefix for renamed files
            batch_size: Optional batch size for processing
            progress_callback: Optional callback for progress updates
        """
        self.api_key = api_key
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        self.metadata_handler = MetadataHandler()
        self.output_handler = OutputHandler()
        self.provider = get_provider(api_key)
        
        # Initialize ExifTool path
        self.exiftool_path = self._find_exiftool()
        self._verify_exiftool()
        
        # Set model and batch size
        self.model = model or self.config.default_model
        self.batch_size = min(batch_size or self.config.default_batch_size, 
                            self.get_max_batch_size(self.model))
        
        # Set file handling options
        self.rename_files = rename_files
        self.prefix = prefix or ''
        
        # Set progress callback
        self.progress_callback = progress_callback
        
        # Processing statistics
        self.stats = {
            'processed': 0,
            'failed': 0,
            'retried': 0,
            'start_time': None,
            'batch_times': [],
            'total_images': 0,
            'current_batch': 0,
            'total_batches': 0,
            'batch_progress': 0,
            'avg_time_per_image': 0,
            'estimated_total_time': None,
            'compression_stats': {
                'enabled': False,
                'total_original_size': 0,
                'total_compressed_size': 0,
                'time_saved': 0
            }
        }
        
        # Initialize console
        self.console = Console()

    def get_max_batch_size(self, model: str) -> int:
        """Get maximum batch size for a given model."""
        # Model-specific batch size limits
        batch_limits = {
            'gemini-2.0-flash-exp': 5,  # Experimental next-gen features
            'gemini-1.5-flash': 5,      # Fast and versatile
            'gemini-1.5-pro': 1,        # Complex reasoning
            'pro': 1,                   # Alias for 1.5-pro
            'default': 3                # Default limit
        }
        return batch_limits.get(model, batch_limits['default'])

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
        """Get the path to the bundled exiftool executable."""
        # Determine platform-specific executable name
        if sys.platform == 'win32':
            exe_name = 'exiftool.exe'
        else:  # Linux/Unix/MacOS
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
        """Verify that exiftool is working correctly."""
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

    def _generate_suggested_filename(self, original_path: Path, metadata: Dict[str, Any]) -> str:
        """
        Generate a suggested filename based on image metadata.
        
        Args:
            original_path: Original image path
            metadata: Image metadata dictionary
            
        Returns:
            Suggested filename with original extension
        """
        components = []
        
        # Add primary subject or first visual element if available
        if metadata.get('visual_elements'):
            components.append(metadata['visual_elements'][0].lower())
        
        # Add mood if available
        if metadata.get('mood'):
            components.append(metadata['mood'].lower())
        
        # Add first composition element if available
        if metadata.get('composition'):
            components.append(metadata['composition'][0].lower())
        
        # Add first keyword if available and not already included
        if metadata.get('keywords'):
            keyword = metadata['keywords'][0].lower()
            if keyword not in components:
                components.append(keyword)
        
        # If we have no components, use a portion of the description
        if not components and metadata.get('description'):
            # Take first few words of description
            desc_words = metadata['description'].split()[:3]
            components.extend(word.lower() for word in desc_words)
        
        # Clean and join components
        clean_components = []
        for component in components:
            # Remove special characters and spaces
            clean = ''.join(c for c in component if c.isalnum() or c.isspace())
            clean = clean.replace(' ', '_')
            if clean:
                clean_components.append(clean)
        
        # Ensure we have at least one component
        if not clean_components:
            clean_components = ['image']
        
        # Join components and add original extension
        suggested_name = '_'.join(clean_components)
        original_extension = original_path.suffix
        
        return f"{suggested_name}{original_extension}"

    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing analysis results
            
        Raises:
            ImageProcessingError: If analysis fails
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                raise ImageValidationError(f"Image file not found: {image_path}")
                
            # Analyze image
            result = await self.provider.analyze_image(image_path)
            
            # Convert string result to dictionary if needed
            if isinstance(result, str):
                result = {"description": result}
            elif not isinstance(result, dict):
                raise ImageProcessingError(f"Invalid result type from provider: {type(result)}")
            
            # Add metadata
            result['original_path'] = image_path
            result['original_filename'] = os.path.basename(image_path)
            result['success'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            raise ImageProcessingError(f"Failed to analyze image: {str(e)}")

    async def analyze_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple images.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of dictionaries containing analysis results
            
        Raises:
            ImageProcessingError: If analysis fails
        """
        try:
            # Validate input
            if not image_paths:
                raise ValueError("No image paths provided")
                
            results = []
            for image_path in image_paths:
                try:
                    result = await self.analyze_image(image_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {image_path}: {str(e)}")
                    results.append({
                        'original_path': image_path,
                        'original_filename': os.path.basename(image_path),
                        'success': False,
                        'error': str(e)
                    })
                    
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing batch: {str(e)}")
            raise ImageProcessingError(f"Failed to analyze batch: {str(e)}")

    def estimate_processing_time(self, total_images: int, compressed: bool = False) -> str:
        """Estimate total processing time based on batch size and model."""
        # Base time estimates (in seconds per image)
        base_times = {
            '1.5-flash': 1.5,
            '2-flash': 2.0,
            'default': 4.0
        }
        
        # Get base time for selected model
        base_time = base_times.get(self.model, base_times['default'])
        
        # Adjust for compression
        if not compressed:
            base_time *= 2.5  # Uncompressed images take longer
            
        total_seconds = total_images * base_time
        
        # Convert to human readable format
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

    def estimate_remaining_time(self) -> float:
        """Estimate remaining processing time based on current statistics."""
        if not self.stats['processed']:
            return 0.0
        
        remaining_images = self.stats['total_images'] - self.stats['processed']
        return remaining_images * self.stats['avg_time_per_image'] 

    async def _process_batch_async(self, batch: List[str], compress: bool = False, progress=None, task=None) -> List[dict]:
        """Process a batch of images asynchronously."""
        try:
            # Create temporary directory for compressed images
            with tempfile.TemporaryDirectory(prefix='llm_image_') as temp_dir:
                # Compress images if needed
                compressed_images = []
                for image_path in batch:
                    if compress:
                        try:
                            compressed_path = os.path.join(temp_dir, f"{os.path.basename(str(image_path))}_compressed.jpg")
                            compress_image(str(image_path), compressed_path)
                            compressed_images.append(compressed_path)
                            if progress and task:
                                progress.update(task, advance=1)
                        except Exception as e:
                            logger.error(f"Error compressing image {image_path}: {str(e)}")
                            compressed_images.append(str(image_path))  # Use original if compression fails
                    else:
                        compressed_images.append(str(image_path))
                        if progress and task:
                            progress.update(task, advance=1)

                # Configure model for structured output
                model = genai.GenerativeModel(model_name=self.model)
                
                # Create generation config
                generation_config = genai.GenerationConfig(
                    temperature=0.1,
                    candidate_count=1,
                    max_output_tokens=2048,
                    top_p=1,
                    top_k=32
                )
                
                # Create safety settings
                safety_settings = {
                    "HARASSMENT": "block_none",
                    "HATE_SPEECH": "block_none",
                    "SEXUALLY_EXPLICIT": "block_none",
                    "DANGEROUS_CONTENT": "block_none"
                }
                
                # Prepare content
                content = [{
                    "text": self.PROMPT
                }]
                
                # Add images to content
                for image_path in compressed_images:
                    with open(image_path, 'rb') as f:
                        content.append({
                            'mime_type': 'image/jpeg',
                            'data': f.read()
                        })
                
                # Generate response
                response = await model.generate_content_async(
                    content,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=True
                )
                
                # Process streaming response
                current_result = ""
                async for chunk in response:
                    if chunk.text:
                        current_result += chunk.text
                        if progress and task:
                            progress.update(task, advance=1)
                
                logger.debug("==================== Starting response parsing ====================")
                logger.debug("Raw response from model:")
                logger.debug(current_result)
                
                try:
                    # Extract content between markers
                    start_marker = current_result.find('<?xml')
                    if start_marker == -1:
                        start_marker = current_result.find('<image_analysis>')
                    
                    end_marker = current_result.find('</image_analysis>')
                    if end_marker == -1:
                        raise ValueError("No valid XML content found in response")
                    
                    xml_content = current_result[start_marker:end_marker + len('</image_analysis>')]
                    
                    # Parse XML content
                    root = etree.fromstring(xml_content.encode('utf-8'))
                    
                    # Extract fields from XML
                    batch_results = {
                        'description': root.findtext('description', ''),
                        'keywords': [kw.text for kw in root.findall('.//keyword') if kw.text],
                        'technical_details': root.findtext('technical_details', ''),
                        'visual_elements': root.findtext('visual_elements', ''),
                        'composition': root.findtext('composition', ''),
                        'mood': root.findtext('mood', ''),
                        'use_cases': root.findtext('use_cases', ''),
                        'suggested_filename': root.findtext('suggested_filename', '')
                    }
                    
                    logger.debug("Successfully parsed XML response to dictionary:")
                    logger.debug(json.dumps(batch_results, indent=2))
                    
                    # Process the results
                    results = []
                    
                    # Since we're getting a single result for multiple images, use it for all
                    for i, image_path in enumerate(batch):
                        processed_result = {
                            'path': str(image_path),
                            'original_path': str(image_path),
                            'original_filename': os.path.basename(str(image_path)),
                            'success': True,
                            'error': ''
                        }
                        
                        # Copy fields directly
                        processed_result.update(batch_results)
                        
                        try:
                            # Write metadata
                            metadata_result = self.write_metadata(processed_result['path'], processed_result)
                            
                            # Update result with metadata operation results
                            processed_result.update({
                                'new_filename': metadata_result.get('new_filename'),
                                'modified_path': metadata_result.get('modified_path'),
                                'success': metadata_result.get('success', False),
                                'error': metadata_result.get('error', '')
                            })
                            
                            if metadata_result.get('success', False):
                                self.stats['processed'] += 1
                                logger.info(f"Successfully processed image {i+1}/{len(batch)}")
                            else:
                                self.stats['failed'] += 1
                                logger.error(f"Failed to process image {i+1}/{len(batch)}: {metadata_result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            logger.error(f"Error writing metadata for image {i+1}/{len(batch)}: {str(e)}")
                            processed_result['success'] = False
                            processed_result['error'] = f'Metadata write error: {str(e)}'
                            self.stats['failed'] += 1
                        
                        results.append(processed_result)
                    
                    return results

                except Exception as e:
                    logger.error(f"Error parsing response: {str(e)}")
                    return [{
                        'path': str(path),
                        'original_path': str(path),
                        'original_filename': os.path.basename(str(path)),
                        'success': False,
                        'error': f'Failed to parse response: {str(e)}'
                    } for path in batch]

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return [{
                'path': str(path),
                'original_path': str(path),
                'original_filename': os.path.basename(str(path)),
                'success': False,
                'error': str(e)
            } for path in batch]

    def _get_image_files(self, folder_path: str) -> List[str]:
        """Get list of image files from folder."""
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder}")
        
        image_files = []
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif']
        
        # Find all image files
        for ext in extensions:
            image_files.extend(str(f) for f in folder.glob(f'*{ext}'))
            image_files.extend(str(f) for f in folder.glob(f'*{ext.upper()}'))
        
        return sorted(image_files)  # Sort for consistent ordering 

    def _parse_batch_results(self, batch: List[Path], result_text: str) -> List[dict]:
        """
        Parse the batch processing results.
        
        Args:
            batch: List of image paths
            result_text: Text output from the model
            
        Returns:
            List of dictionaries containing parsed results
        """
        results = []
        current_result = {}
        current_field = None
        
        try:
            # Split text into lines and process each line
            lines = result_text.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip empty lines
                if not line:
                    i += 1
                    continue
                
                # Check for new image analysis section
                if "Analysis" in line and ("Image" in line or "**" in line):
                    if current_result:
                        # Add the path information
                        if len(results) < len(batch):
                            current_result['path'] = str(batch[len(results)].resolve())
                            current_result['success'] = True
                        results.append(current_result)
                    current_result = {}
                    current_field = None
                    i += 1
                    continue
                
                # Check for field headers
                if "**Description:**" in line or line.startswith("Description:"):
                    current_field = 'description'
                    value = line.split(':', 1)[1].strip() if ':' in line else ''
                    current_result[current_field] = value
                elif "**Keywords:**" in line or line.startswith("Keywords:"):
                    current_field = 'keywords'
                    value = line.split(':', 1)[1].strip() if ':' in line else ''
                    current_result[current_field] = [k.strip() for k in value.split(',') if k.strip()]
                elif "**Technical Details:**" in line or line.startswith("Technical Details:"):
                    current_field = 'technical_details'
                    current_result[current_field] = {}
                    # Parse technical details
                    if ':' in line:
                        tech_text = line.split(':', 1)[1].strip()
                        if 'JPEG' in tech_text:
                            current_result[current_field]['format'] = 'JPEG'
                        if 'dimensions' in tech_text.lower():
                            dims = tech_text.split('dimensions')[1].split(',')[0].strip()
                            current_result[current_field]['dimensions'] = dims
                        if 'color space' in tech_text.lower():
                            cs = tech_text.split('color space')[1].split(',')[0].strip()
                            current_result[current_field]['color_space'] = cs
                elif "**Visual Elements:**" in line or line.startswith("Visual Elements:"):
                    current_field = 'visual_elements'
                    current_result[current_field] = []
                elif "**Composition:**" in line or line.startswith("Composition:"):
                    current_field = 'composition'
                    value = line.split(':', 1)[1].strip() if ':' in line else ''
                    current_result[current_field] = value
                elif "**Mood:**" in line or line.startswith("Mood:"):
                    current_field = 'mood'
                    value = line.split(':', 1)[1].strip() if ':' in line else ''
                    current_result[current_field] = value
                elif "**Use Cases:**" in line or line.startswith("Use Cases:"):
                    current_field = 'use_cases'
                    value = line.split(':', 1)[1].strip() if ':' in line else ''
                    current_result[current_field] = [u.strip() for u in value.split(',') if u.strip()]
                elif "**Suggested Filename:**" in line or line.startswith("Suggested Filename:"):
                    current_field = 'suggested_filename'
                    value = line.split(':', 1)[1].strip() if ':' in line else ''
                    current_result[current_field] = value
                else:
                    # Add content to current field if we have one
                    if current_field == 'visual_elements':
                        # Parse bullet points
                        if line.strip('* '):
                            items = line.strip('* ').split(',')
                            current_result[current_field].extend([item.strip() for item in items if item.strip()])
                    elif current_field and current_field in current_result:
                        # Append to existing content
                        if isinstance(current_result[current_field], list):
                            current_result[current_field].extend([item.strip() for item in line.split(',') if item.strip()])
                        elif isinstance(current_result[current_field], str):
                            current_result[current_field] = f"{current_result[current_field]} {line}".strip()
                
                i += 1
            
            # Add the last result if it exists
            if current_result:
                if len(results) < len(batch):
                    current_result['path'] = str(batch[len(results)].resolve())
                    current_result['success'] = True
                results.append(current_result)
            
            # Ensure we have a result for each image
            while len(results) < len(batch):
                results.append({
                    'path': str(batch[len(results)].resolve()),
                    'success': False,
                    'error': 'No analysis generated'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error parsing batch results: {str(e)}")
            # Return error results for all images
            return [{
                'path': str(path.resolve()),
                'success': False,
                'error': f'Failed to parse results: {str(e)}'
            } for path in batch]

    def _update_result_field(self, result: Dict[str, Any], field: str, content: List[str]) -> None:
        """Helper method to update result dictionary with parsed field content."""
        # Clean field name
        field = field.strip('*').strip()
        field = re.sub(r'\s+', '_', field)
        
        # Join content
        content_text = ' '.join(content).strip()
        
        # Handle special fields
        if field.lower() == 'technical_details':
            # Parse technical details
            for line in content:
                if 'Format:' in line:
                    result['Technical_format'] = line.split('Format:', 1)[1].strip()
                elif 'dimensions' in line.lower():
                    result['Technical_dimensions'] = line
                elif 'Color Space:' in line:
                    result['Technical_color_space'] = line.split('Color Space:', 1)[1].strip()
        elif field.lower() in ['keywords', 'visual_elements', 'composition', 'use_cases']:
            # Split by commas and clean
            items = content_text.replace('*', '').split(',')
            result[field] = [item.strip() for item in items if item.strip()]
        else:
            # Regular fields
            result[field] = content_text
            
        return result

    def write_metadata(self, image_path: Union[str, Path], metadata: Dict[str, Any], backup: bool = True, duplicate: bool = False) -> Dict[str, Any]:
        """Write metadata to an image file."""
        try:
            # Convert metadata to standard EXIF/IPTC/XMP tags
            exif_metadata = {}
            
            # Map our fields to standard metadata tags
            if isinstance(metadata.get('description'), str):
                exif_metadata['Description'] = metadata['description'].strip()
            if isinstance(metadata.get('keywords'), (list, tuple)):
                exif_metadata['Keywords'] = [k.strip() for k in metadata['keywords'] if k.strip()]
            if isinstance(metadata.get('visual_elements'), (list, tuple)):
                exif_metadata['VisualElements'] = [v.strip() for v in metadata['visual_elements'] if v.strip()]
            if isinstance(metadata.get('composition'), (list, tuple)):
                exif_metadata['Composition'] = [c.strip() for c in metadata['composition'] if c.strip()]
            if isinstance(metadata.get('mood'), str):
                exif_metadata['Mood'] = metadata['mood'].strip()
            if isinstance(metadata.get('use_cases'), (list, tuple)):
                exif_metadata['UseCases'] = [u.strip() for u in metadata['use_cases'] if u.strip()]
            
            # Add technical details if present
            if isinstance(metadata.get('technical_details'), dict):
                tech = metadata['technical_details']
                tech_details = []
                if 'format' in tech:
                    exif_metadata['FileType'] = tech['format'].strip()
                    tech_details.append(f"Format: {tech['format']}")
                if 'dimensions' in tech:
                    exif_metadata['ImageSize'] = tech['dimensions'].strip()
                    tech_details.append(f"Dimensions: {tech['dimensions']}")
                if 'color_space' in tech:
                    exif_metadata['ColorSpace'] = tech['color_space'].strip()
                    tech_details.append(f"Color Space: {tech['color_space']}")
                if tech_details:
                    exif_metadata['TechnicalDetails'] = '; '.join(tech_details)
            
            # Add software tag
            exif_metadata['Software'] = 'Image Sense AI Processor'
            
            # Add title if available
            if isinstance(metadata.get('title'), str):
                exif_metadata['Title'] = metadata['title'].strip()
            
            # Add suggested filename if available
            if isinstance(metadata.get('suggested_filename'), str):
                exif_metadata['Label'] = metadata['suggested_filename'].strip()
            
            # Write metadata using ExifTool
            result = {
                'original_path': str(image_path),
                'original_filename': os.path.basename(str(image_path)),
                'success': True
            }
            
            # Write the metadata
            metadata_handler = MetadataHandler(self.exiftool_path)
            metadata_result = metadata_handler.write_metadata(
                image_path=image_path,
                metadata=exif_metadata,
                backup=backup,
                duplicate=duplicate
            )
            
            # Update result with metadata operation results
            result.update(metadata_result)
            return result
            
        except Exception as e:
            logger.error(f"Error writing metadata: {str(e)}")
            return {
                'original_path': str(image_path),
                'original_filename': os.path.basename(str(image_path)),
                'success': False,
                'error': str(e)
            } 

    async def process_images(self, input_path: Union[str, List[str]], output_dir: str = None, compress: bool = None, verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Process one or more images, either from a directory or a list of file paths.
        
        Args:
            input_path: Either a directory path or a list of image file paths
            output_dir: Optional output directory for results
            compress: Whether to compress images before processing
            verbose: Whether to show detailed progress
            
        Returns:
            List of dictionaries containing results for each processed image
        """
        try:
            if isinstance(input_path, str):
                # Input is a directory path
                input_path = Path(input_path)
                if not input_path.exists():
                    raise ValueError(f"Input path does not exist: {input_path}")
                
                if input_path.is_dir():
                    # Create output directory if it doesn't exist
                    output_dir = input_path / 'output'
                    output_dir.mkdir(exist_ok=True)
                    
                    # Get all image files in directory
                    image_paths = []
                    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif']:
                        image_paths.extend(input_path.glob(f"*{ext}"))
                    
                    if not image_paths:
                        raise ValueError(f"No image files found in directory: {input_path}")
                    
                    return await self.process_batch(image_paths, output_dir, compress=compress, verbose=verbose)
                else:
                    # Single file path
                    output_dir = input_path.parent / 'output'
                    output_dir.mkdir(exist_ok=True)
                    return await self.process_batch([input_path], output_dir, compress=compress, verbose=verbose)
            else:
                # Input is a list of file paths
                if not input_path:
                    raise ValueError("Empty list of image paths provided")
                
                # Use the parent directory of the first image as output directory
                first_path = Path(input_path[0])
                output_dir = first_path.parent / 'output'
                output_dir.mkdir(exist_ok=True)
                
                # Convert all paths to Path objects
                image_paths = [Path(p) for p in input_path]
                return await self.process_batch(image_paths, output_dir, compress=compress, verbose=verbose)
                
        except Exception as e:
            logger.error(f"Error processing images: {str(e)}")
            raise ImageProcessingError(f"Failed to process images: {str(e)}")

    def _display_config_summary(self):
        """Display a summary of the current configuration settings."""
        self.console.print("\n[bold blue]Configuration Summary:[/bold blue]")
        self.console.print(f"├── Model: {self.model}")
        self.console.print(f"├── Batch Size: {self.batch_size}")
        self.console.print(f"├── File Renaming: {'Enabled' if self.rename_files else 'Disabled'}")
        self.console.print(f"├── Prefix: {self.prefix or 'None'}")
        self.console.print(f"└── ExifTool Path: {self.exiftool_path}\n")

    async def process_batch(self, image_paths: List[Path], output_dir: Optional[Path] = None, compress: bool = None, verbose: bool = False) -> List[Dict[str, Any]]:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of paths to image files
            output_dir: Optional directory to save processed results
            compress: Whether to compress images before processing
            verbose: Whether to show detailed progress
            
        Returns:
            List of dictionaries containing results for each processed image
        """
        try:
            # Create temporary output directory if none provided
            if output_dir is None:
                output_dir = Path(tempfile.mkdtemp())

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process images
            results = []
            for image_path in image_paths:
                try:
                    # Analyze image
                    result = await self.analyze_image(str(image_path))
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {str(e)}")
                    results.append({
                        'original_path': str(image_path),
                        'original_filename': image_path.name,
                        'success': False,
                        'error': str(e)
                    })

            return results

        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")
            raise ImageProcessingError(f"Failed to process batch: {str(e)}")

    def _compress_image(self, image_path: str) -> Optional[str]:
        """Compress an image for LLM processing."""
        try:
            from PIL import Image
            import os
            
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(image_path), 'compressed')
            os.makedirs(output_dir, exist_ok=True)
            
            # Open and compress image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate new dimensions while maintaining aspect ratio
                max_size = max(img.size)
                ratio = self.config.max_dimension / max_size if max_size > self.config.max_dimension else 1.0
                new_size = tuple(int(dim * ratio) for dim in img.size)
                
                # Resize and compress
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                compressed_path = os.path.join(output_dir, f"compressed_{os.path.basename(image_path)}")
                img.save(compressed_path, 'JPEG', quality=self.config.compression_quality, optimize=True)
                
                return compressed_path
                
        except Exception as e:
            self.logger.error(f"Error compressing image: {str(e)}")
            return None 

    def save_to_csv(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to CSV file."""
        import pandas as pd
        
        # Flatten nested structures for CSV
        flattened_results = []
        for result in results:
            flat_result = {
                'original_path': result.get('original_path', ''),
                'original_filename': result.get('original_filename', ''),
                'success': result.get('success', False),
                'error': result.get('error', ''),
            }
            
            # Only add analysis fields if successful
            if result.get('success', False):
                # Add description
                flat_result['description'] = result.get('description', '')
                
                # Add technical details
                tech = result.get('technical_details', {})
                if isinstance(tech, dict):
                    flat_result.update({
                        'format': tech.get('format', ''),
                        'dimensions': tech.get('dimensions', ''),
                        'color_space': tech.get('color_space', '')
                    })
                
                # Handle list fields by joining with semicolons
                for field in ['keywords', 'visual_elements', 'composition', 'use_cases']:
                    values = result.get(field, [])
                    if isinstance(values, (list, tuple)):
                        flat_result[field] = '; '.join(str(v).strip() for v in values if v)
                    else:
                        flat_result[field] = str(values).strip()
                
                # Add mood and suggested filename
                flat_result['mood'] = str(result.get('mood', '')).strip()
                flat_result['suggested_filename'] = str(result.get('suggested_filename', '')).strip()
            
            flattened_results.append(flat_result)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(flattened_results)
        
        # Reorder columns for better readability
        column_order = [
            'original_path',
            'original_filename',
            'success',
            'error',
            'description',
            'keywords',
            'format',
            'dimensions',
            'color_space',
            'visual_elements',
            'composition',
            'mood',
            'use_cases',
            'suggested_filename'
        ]
        
        # Only include columns that exist in the DataFrame
        columns = [col for col in column_order if col in df.columns]
        df = df[columns]
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to CSV: {output_path}")

    def save_to_xml(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """Save results to XML file."""
        try:
            # Create root element
            root = etree.Element("image_analysis_results")
            
            # Add each image result
            for result in results:
                image = etree.SubElement(root, "image")
                
                # Add path information
                path_info = etree.SubElement(image, "path_info")
                for field in ['original_path', 'original_filename', 'new_filename', 'modified_path']:
                    if result.get(field):
                        elem = etree.SubElement(path_info, field)
                        elem.text = str(result[field])
                
                # Add status
                status = etree.SubElement(image, "status")
                success = etree.SubElement(status, "success")
                success.text = str(result.get('success', False))
                if not result.get('success', False):
                    error = etree.SubElement(status, "error")
                    error.text = result.get('error', '')
                    continue
                
                # Add analysis results
                if result.get('description'):
                    description = etree.SubElement(image, "description")
                    description.text = result['description']
                
                if result.get('keywords'):
                    keywords = etree.SubElement(image, "keywords")
                    for kw in result['keywords']:
                        if kw:
                            keyword = etree.SubElement(keywords, "keyword")
                            keyword.text = str(kw)
                
                if result.get('technical_details'):
                    technical = etree.SubElement(image, "technical_details")
                    if isinstance(result['technical_details'], dict):
                        for key, value in result['technical_details'].items():
                            if value:
                                detail = etree.SubElement(technical, key)
                                detail.text = str(value)
                    else:
                        technical.text = str(result['technical_details'])
                
                if result.get('visual_elements'):
                    visual = etree.SubElement(image, "visual_elements")
                    if isinstance(result['visual_elements'], list):
                        for elem in result['visual_elements']:
                            if elem:
                                element = etree.SubElement(visual, "element")
                                element.text = str(elem)
                    else:
                        visual.text = str(result['visual_elements'])
                
                if result.get('composition'):
                    composition = etree.SubElement(image, "composition")
                    if isinstance(result['composition'], list):
                        for comp in result['composition']:
                            if comp:
                                technique = etree.SubElement(composition, "technique")
                                technique.text = str(comp)
                    else:
                        composition.text = str(result['composition'])
                
                if result.get('mood'):
                    mood = etree.SubElement(image, "mood")
                    mood.text = str(result['mood'])
                
                if result.get('use_cases'):
                    use_cases = etree.SubElement(image, "use_cases")
                    if isinstance(result['use_cases'], list):
                        for use in result['use_cases']:
                            if use:
                                use_case = etree.SubElement(use_cases, "use_case")
                                use_case.text = str(use)
                    else:
                        use_cases.text = str(result['use_cases'])
                
                if result.get('suggested_filename'):
                    filename = etree.SubElement(image, "suggested_filename")
                    filename.text = str(result['suggested_filename'])
            
            # Write to file with pretty formatting
            tree = etree.ElementTree(root)
            tree.write(output_path, pretty_print=True, xml_declaration=True, encoding='utf-8')
            
            self.logger.info(f"Saved results to XML: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving XML: {str(e)}")
            raise

    def display_banner(self):
        """Display the application banner."""
        console = Console()
        console.print(BANNER)