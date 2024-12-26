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
from typing import List, Dict, Any, Optional, Union, Callable
from typing_extensions import TypedDict
import google.generativeai as genai
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeRemainingColumn, TransferSpeedColumn
)
from lxml import etree
import shutil
import absl.logging
import asyncio
from PIL import Image
import numpy as np
import glob
import pandas as pd

from .image_utils import compress_image, create_llm_optimized_copy
from .llm_handler import get_provider
from .metadata_handler import MetadataHandler, MetadataError
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

class TechnicalDetails(TypedDict):
    """Technical details of an image."""
    format: str
    dimensions: str
    color_space: str

class ImageAnalysis(TypedDict):
    """Full image analysis results."""
    description: str
    keywords: List[str]
    technical_details: TechnicalDetails
    visual_elements: List[str]
    composition: List[str]
    mood: str
    use_cases: List[str]
    suggested_filename: str

class ImageProcessingError(Exception):
    """Exception raised when image processing fails."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error

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
        '1.5-flash': 'gemini-1.5-flash',       # Fast and versatile
        '1.5-flash-8b': 'gemini-1.5-flash-8b', # High volume, lower intelligence
        '1.5-pro': 'gemini-1.5-pro',           # Complex reasoning
        'pro': 'gemini-1.5-pro',               # Alias for 1.5-pro
    }

    def __init__(self, api_key: str, config: Optional[Config] = None, model: Optional[str] = None, 
                 rename_files: bool = False, prefix: Optional[str] = None, batch_size: Optional[int] = None,
                 progress_callback: Optional[Callable] = None, verbose_output: Optional[bool] = None):
        """Initialize the image processor.
        
        Args:
            api_key: API key for the service
            config: Optional configuration object
            model: Optional model name to use
            rename_files: Whether to rename processed files
            prefix: Optional prefix for renamed files
            batch_size: Optional batch size for processing
            progress_callback: Optional callback for progress updates
            verbose_output: Optional override for verbose output setting
        """
        self.api_key = api_key
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        
        # Set logging level from environment
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.logger.setLevel(log_level)
        
        # Initialize handlers
        self.metadata_handler = MetadataHandler()
        self.output_handler = OutputHandler()
        
        # Initialize provider and model
        self.provider = get_provider(api_key, verbose=verbose_output)
        genai.configure(api_key=api_key)
        
        # Set model name from environment or parameter
        if model in self.AVAILABLE_MODELS:
            self.model = self.AVAILABLE_MODELS[model]
        elif model in self.AVAILABLE_MODELS.values():
            self.model = model
        else:
            self.model = os.getenv('DEFAULT_MODEL', 'gemini-2.0-flash-exp')
        
        # Set batch size based on model and environment
        default_batch = int(os.getenv('DEFAULT_BATCH_SIZE', '8'))
        max_batch = int(os.getenv('MAX_BATCH_SIZE', '16'))
        self.batch_size = min(
            batch_size or default_batch,
            max_batch,
            self.get_max_batch_size(self.model)
        )
        
        # Set file handling options from environment
        self.rename_files = rename_files or os.getenv('RENAME_FILES', 'false').lower() == 'true'
        self.prefix = prefix or os.getenv('FILE_PREFIX', '')
        
        # Set progress callback and verbose output
        self.progress_callback = progress_callback
        self.verbose_output = verbose_output if verbose_output is not None else os.getenv('VERBOSE_OUTPUT', 'true').lower() == 'true'
        
        # Set compression settings from environment
        self.compression_enabled = os.getenv('COMPRESSION_ENABLED', 'true').lower() == 'true'
        self.compression_quality = int(os.getenv('COMPRESSION_QUALITY', '85'))
        self.max_dimension = int(os.getenv('MAX_DIMENSION', '1024'))
        
        # Set metadata settings from environment
        self.backup_metadata = os.getenv('BACKUP_METADATA', 'true').lower() == 'true'
        self.write_exif = os.getenv('WRITE_EXIF', 'true').lower() == 'true'
        self.duplicate_files = os.getenv('DUPLICATE_FILES', 'false').lower() == 'true'
        self.duplicate_suffix = os.getenv('DUPLICATE_SUFFIX', '_modified')
        
        # Set output settings
        self.output_format = os.getenv('DEFAULT_OUTPUT_FORMAT', 'csv')
        self.output_directory = os.getenv('OUTPUT_DIRECTORY', 'output')
        
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
            if self.verbose_output:
                self.console.print(f"\n[cyan]🖼️ Analyzing image:[/] {image_path}")
            
            # Validate image path
            if not os.path.exists(image_path):
                raise ImageProcessingError(f"Image file not found: {image_path}")
            
            # Start timing
            start_time = time.time()
            
            # Get image info first
            image_info = await self._get_image_info(image_path)
            
            # Analyze image
            result = await self._analyze_image_async(image_path)
            
            # Add metadata
            result.update({
                'original_path': image_path,
                'original_filename': os.path.basename(image_path),
                'technical_details': {
                    'format': image_info['format'],
                    'dimensions': f"{image_info['size'][0]}x{image_info['size'][1]}",
                    'color_space': image_info['mode']
                }
            })
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            if self.verbose_output:
                self.console.print(f"[green]✓[/] Analysis complete ({processing_time:.2f}s)")
                self.console.print("\n[cyan]📝 Analysis Results:[/]")
                self._display_analysis_results(result)
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to analyze image: {str(e)}"
            if self.verbose_output:
                self.console.print(f"[red]✗ {error_msg}[/]")
            logger.error(error_msg)
            raise ImageProcessingError(error_msg, original_error=e)

    async def _analyze_image_async(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a single image asynchronously.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            if self.verbose_output:
                with self.console.status("[cyan]Getting image analysis...[/]"):
                    response = await self.provider.generate_content(image_path)
            else:
                response = await self.provider.generate_content(image_path)
            
            # Parse response
            if not response or not response.get('content'):
                raise ImageProcessingError("No valid response from provider")
            
            return response
            
        except Exception as e:
            error_msg = f"Error in async image analysis: {str(e)}"
            logger.error(error_msg)
            raise ImageProcessingError(error_msg, original_error=e)

    def _display_analysis_results(self, result: Dict[str, Any]) -> None:
        """Display analysis results in a formatted way."""
        try:
            # Create a table for results
            table = Table(box=box.ROUNDED)
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            
            # Add basic info
            table.add_row("Filename", result['original_filename'])
            table.add_row("Format", result['technical_details']['format'])
            table.add_row("Dimensions", result['technical_details']['dimensions'])
            
            # Add description if available
            if 'description' in result:
                table.add_row("Description", result['description'])
            
            # Add keywords if available
            if 'keywords' in result:
                table.add_row("Keywords", ", ".join(result['keywords']))
            
            # Add mood if available
            if 'mood' in result:
                table.add_row("Mood", result['mood'])
            
            # Add suggested filename if available
            if 'suggested_filename' in result:
                table.add_row("Suggested Filename", result['suggested_filename'])
            
            self.console.print(table)
            
        except Exception as e:
            logger.warning(f"Error displaying results: {str(e)}")
            # Fall back to simple print if table fails
            self.console.print(result)

    async def _get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Get image information asynchronously."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_image_info_sync, image_path)
        except Exception as e:
            error_msg = f"Error getting image info: {str(e)}"
            logger.error(error_msg)
            raise ImageProcessingError(error_msg, original_error=e)

    def _get_image_info_sync(self, image_path: str) -> Dict[str, Any]:
        """Get image information synchronously."""
        try:
            with Image.open(image_path) as img:
                return {
                    'format': img.format or 'Unknown',
                    'mode': img.mode or 'Unknown',
                    'size': img.size,
                    'info': img.info
                }
        except Exception as e:
            error_msg = f"Error reading image file: {str(e)}"
            logger.error(error_msg)
            raise ImageProcessingError(error_msg, original_error=e)

    def _parse_xml_content(self, xml_content: str) -> Dict[str, Any]:
        """Parse XML content from LLM response.
        
        Args:
            xml_content: XML string from LLM
            
        Returns:
            Dictionary containing parsed fields
            
        Raises:
            ValueError: If XML parsing fails
        """
        try:
            # Remove XML declaration if present
            if xml_content.startswith('<?xml'):
                xml_content = xml_content[xml_content.find('?>')+2:].strip()
            
            # Parse XML
            root = etree.fromstring(xml_content.encode('utf-8'))
            
            # Helper function to safely extract text
            def get_text(element, xpath: str, default: str = '') -> str:
                node = element.find(xpath)
                return node.text.strip() if node is not None and node.text else default
            
            # Helper function to safely extract list
            def get_list(element, xpath: str) -> List[str]:
                return [e.text.strip() for e in element.findall(xpath) if e is not None and e.text]
            
            # Extract all fields with validation
            result = {
                'description': get_text(root, 'description'),
                'keywords': get_list(root, './/keyword'),
                'technical_details': {
                    'format': get_text(root, './/technical_details/format', 'Unknown'),
                    'dimensions': get_text(root, './/technical_details/dimensions', 'Unknown'),
                    'color_space': get_text(root, './/technical_details/color_space', 'Unknown')
                },
                'visual_elements': [get_text(root, 'visual_elements')] if get_text(root, 'visual_elements') else [],
                'composition': [get_text(root, 'composition')] if get_text(root, 'composition') else [],
                'mood': get_text(root, 'mood'),
                'use_cases': [use_case.strip() for use_case in get_text(root, 'use_cases').split(',') if use_case.strip()],
                'suggested_filename': get_text(root, 'suggested_filename')
            }
            
            # Validate required fields
            required_fields = ['description', 'keywords']
            missing_fields = [field for field in required_fields if not result.get(field)]
            
            if missing_fields:
                raise ValueError(f"Missing required fields in XML: {', '.join(missing_fields)}")
                
            return result
            
        except etree.ParseError as e:
            raise ValueError(f"Invalid XML format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error parsing XML content: {str(e)}")

    async def process_batch(self, image_paths: List[str], output_dir: str = None) -> Dict[str, Any]:
        """Process a batch of images and return their analysis results."""
        if not image_paths:
            return {"success": False, "error": "No images provided"}

        results = []
        successful_count = 0

        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = os.path.dirname(image_paths[0])

        # Generate CSV filename based on input folder name
        input_folder_name = os.path.basename(os.path.dirname(image_paths[0]))
        csv_filename = os.path.join(output_dir, f"{input_folder_name}_results.csv")

        if self.verbose_output:
            console.print("\n[bold cyan]🎯 Starting Batch Processing[/]")
            console.print(f"[dim]Total images to process: {len(image_paths)}[/]")

        batch_size = min(100, len(image_paths))  # Process up to 100 images at once

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TransferSpeedColumn(),
            console=console
        ) as progress:
            overall_progress = progress.add_task(
                "[bold blue]Overall Progress", 
                total=len(image_paths)
            )

            # Process images in batches
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i+batch_size]
                retries = 0
                retry_count = 3
                success = False

                while retries < retry_count and not success:
                    try:
                        if self.verbose_output:
                            console.print(f"\n[bold cyan]📸 Processing Batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}[/]")
                            console.print(f"[dim]Processing {len(batch)} images[/]")

                        # Process batch using provider
                        batch_response = await self.provider.process_batch(batch)

                        if not batch_response:
                            raise ValueError("No response from provider")

                        # Process each image result
                        for image_result in batch_response:
                            path = image_result['path']
                            content = image_result['content']

                            try:
                                # Get image info
                                image_info = await self._get_image_info(path)

                                # Prepare result
                                result = {
                                    'original_path': path,
                                    'original_filename': os.path.basename(path),
                                    'success': True,
                                    'description': content['description'],
                                    'keywords': content['keywords'],
                                    'visual_elements': content['visual_elements'],
                                    'mood': content['mood'],
                                    'use_cases': content['use_cases'],
                                    'format': image_info.get('format', 'Unknown'),
                                    'dimensions': f"{image_info.get('size', (0, 0))[0]}x{image_info.get('size', (0, 0))[1]}",
                                    'color_space': image_info.get('mode', 'Unknown')
                                }

                                # Write metadata if enabled
                                if self.write_exif:
                                    exif_metadata = {
                                        'technical_details': {
                                            'format': image_info.get('format', 'Unknown'),
                                            'dimensions': f"{image_info.get('size', (0, 0))[0]}x{image_info.get('size', (0, 0))[1]}",
                                            'color_space': image_info.get('mode', 'Unknown')
                                        },
                                        'description': content['description'],
                                        'keywords': content['keywords'],
                                        'visual_elements': content['visual_elements'],
                                        'mood': content['mood'],
                                        'use_cases': content['use_cases']
                                    }

                                    write_result = self.metadata_handler.write_metadata(
                                        path,
                                        exif_metadata,
                                        backup=self.backup_metadata,
                                        duplicate=self.duplicate_files
                                    )

                                    if write_result.get('success'):
                                        if self.verbose_output:
                                            console.print(f"[green]✓ {write_result.get('message', 'EXIF metadata written successfully')}[/]")
                                    else:
                                        error_msg = write_result.get('error', 'Unknown error')
                                        if self.verbose_output:
                                            console.print(f"[red]✗ Failed to write EXIF metadata: {error_msg}[/]")
                                        logger.error(f"EXIF write failed: {error_msg}")

                                results.append(result)

                            except Exception as e:
                                logger.error(f"Error processing metadata for {path}: {str(e)}")
                                results.append({
                                    'original_path': path,
                                    'original_filename': os.path.basename(path),
                                    'success': False,
                                    'error': f"Metadata processing error: {str(e)}"
                                })

                        success = True

                    except Exception as e:
                        retries += 1
                        if retries >= retry_count:
                            if self.verbose_output:
                                console.print(f"[red]✗ Failed after {retry_count} attempts:[/] {str(e)}")

                            for path in batch:
                                results.append({
                                    'original_path': path,
                                    'original_filename': os.path.basename(path),
                                    'success': False,
                                    'error': str(e)
                                })
                        else:
                            if self.verbose_output:
                                console.print(f"[yellow]⚠️ Error occurred, retrying... ({retries}/{retry_count})[/]")
                            await asyncio.sleep(1)  # Wait before retry

                    finally:
                        progress.update(overall_progress, advance=len(batch))

        if self.verbose_output:
            # Print final summary
            success = len([r for r in results if isinstance(r, dict) and r.get('success', False)])
            failed = len([r for r in results if isinstance(r, dict) and not r.get('success', False)])
            console.print("\n[bold cyan]📊 Processing Summary[/]")
            console.print(f"[green]✓ Successfully processed: {success} images[/]")
            if failed > 0:
                console.print(f"[red]✗ Failed to process: {failed} images[/]")

        # Write results to CSV
        try:
            df = pd.DataFrame(results)
            df.to_csv(csv_filename, index=False)
            logger.info(f"Saved results to CSV: {csv_filename}")
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")

        # Clean up ExifTool backup files
        try:
            backup_files = glob.glob(os.path.join(os.path.dirname(image_paths[0]), "*_original"))
            for backup_file in backup_files:
                os.remove(backup_file)
            logger.info(f"Cleaned up {len(backup_files)} ExifTool backup files")
        except Exception as e:
            logger.error(f"Error cleaning up backup files: {e}")

        return {
            "success": True,
            "processed_count": successful_count,
            "total_count": len(image_paths),
            "csv_path": csv_filename
        }

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
                'error': result.get('error', '')
            }
            
            # Only add analysis fields if successful
            if result.get('success', True):
                # Add description and mood
                flat_result['description'] = result.get('description', '')
                flat_result['mood'] = result.get('mood', '')
                
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
                    if isinstance(values, list):
                        flat_result[field] = '; '.join(str(v).strip() for v in values if v)
                    else:
                        flat_result[field] = str(values).strip()
                
                # Add suggested filename
                flat_result['suggested_filename'] = result.get('suggested_filename', '')
            
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
        if self.verbose_output:
            console.print(f"[green]✓ Results saved to CSV:[/] {output_path}")
            # Print a preview of the CSV
            console.print("\n[cyan]📄 CSV Preview:[/]")
            console.print(df.head().to_string())
        logger.info(f"Saved results to CSV: {output_path}")

    def save_to_xml(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """Save results to XML file."""
        try:
            # Create root element
            root = etree.Element("image_analysis_results")
            
            # Create XML declaration separately
            doc = etree.ElementTree(root)
            doc.write(output_path, encoding="UTF-8", xml_declaration=True, pretty_print=True)
            
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
                success.text = str(result.get('success', False)).lower()
                if not result.get('success', False):
                    error = etree.SubElement(status, "error")
                    error.text = result.get('error', '')
                    continue
                
                # Add analysis results
                if result.get('description'):
                    description = etree.SubElement(image, "description")
                    description.text = result['description']
                
                # Add keywords section
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
            
            # Write the XML file with proper formatting
            doc.write(output_path, encoding="UTF-8", xml_declaration=True, pretty_print=True)
            logger.info(f"Saved results to XML: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving XML: {str(e)}")
            raise

    def display_banner(self):
        """Display the application banner."""
        console = Console()
        console.print(BANNER)