"""LLM provider module for image analysis."""
import os
import logging
import tempfile
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TransferSpeedColumn
from rich import box
from rich.table import Table
from pathlib import Path
import shutil
import absl.logging
from . import image_utils
from .config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prevent absl from writing to stderr
logging.root.removeHandler(absl.logging._absl_handler)

# Rich console for pretty output
console = Console()

# Create config instance
config = Config()

# Content template for image analysis
CONTENT_TEMPLATE = """Analyze this image and provide detailed information in XML format. Follow these guidelines:
1. Include a natural description of the image
2. Extract key visual elements and composition techniques
3. Note any technical details visible in the image
4. Suggest relevant use cases for the image
5. Make the suggested filename descriptive and use underscores between words
6. Keep all text content clear and concise

Example format:
<?xml version="1.0" encoding="UTF-8"?>
<image_analysis>
    <original_filename>{filename}</original_filename>
    <description>A clear, concise description of the main subject and scene</description>
    <keywords>
        <keyword>key term 1</keyword>
        <keyword>key term 2</keyword>
    </keywords>
    <technical_details>Technical aspects of the image</technical_details>
    <visual_elements>Notable visual elements and their arrangement</visual_elements>
    <composition>Description of composition techniques used</composition>
    <mood>Overall mood or emotional impact</mood>
    <use_cases>Potential applications or use cases</use_cases>
    <suggested_filename>descriptive_filename.jpg</suggested_filename>
</image_analysis>"""

class GeminiProvider:
    """Provider class for Google's Gemini Vision API."""
    
    def __init__(self, api_key: str, model: Optional[str] = None, verbose: bool = False):
        """Initialize the Gemini provider."""
        self.api_key = api_key
        self.verbose = verbose
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Get config instance
        config = Config()
        
        # Set model name
        logger.debug(f"Initializing GeminiProvider with model parameter: {model}")
        logger.debug(f"Config default_model: {config.default_model}")
        self.model_name = model or config.default_model
        logger.debug(f"Selected model_name: {self.model_name}")
        
        try:
            self.model = genai.GenerativeModel(self.model_name)
            if self.verbose:
                console.print(f"[green]✓[/] Initialized Gemini model: {self.model_name}")
        except Exception as e:
            if self.verbose:
                console.print(f"[red]✗ Error initializing Gemini model:[/] {str(e)}")
            raise Exception(f"Failed to initialize Gemini model: {str(e)}")
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image file."""
        try:
            return Image.open(image_path)
        except Exception as e:
            if self.verbose:
                console.print(f"[red]✗ Error loading image:[/] {str(e)}")
            raise Exception(f"Failed to load image: {str(e)}")

    async def generate_content(self, image_path: str) -> Dict[str, Any]:
        """Generate content for an image using Gemini."""
        try:
            if self.verbose:
                with console.status("[bold blue]🤖 Processing with Gemini...[/]") as status:
                    # Get original filename
                    original_filename = os.path.basename(image_path)
                    
                    # Compress image for LLM
                    status.update("[bold yellow]🔄 Optimizing image for Gemini...[/]")
                    original_size = os.path.getsize(image_path) / 1024  # KB
                    
                    temp_dir, compressed_path = image_utils.create_llm_optimized_copy(
                        image_path,
                        max_dimension=1024,  # Optimize for Gemini
                        quality=85
                    )
                    
                    compressed_size = os.path.getsize(compressed_path) / 1024  # KB
                    reduction = (1 - compressed_size/original_size) * 100
                    
                    console.print(f"[green]✓ Image optimized[/]")
                    console.print(f"[dim]Original size: {original_size:.1f}KB[/]")
                    console.print(f"[dim]Optimized size: {compressed_size:.1f}KB[/]")
                    console.print(f"[dim]Reduction: {reduction:.1f}%[/]")
                    
                    # Load and prepare the image
                    status.update("[bold yellow]📸 Loading image...[/]")
                    img = self._load_image(compressed_path)
                    
                    # Create the prompt parts and generate content
                    status.update("[bold yellow]🚀 Sending to Gemini API...[/]")
                    console.print("[dim]Waiting for response from Gemini...[/]")
                    
                    # Format template with filename
                    formatted_template = CONTENT_TEMPLATE.format(filename=original_filename)
                    
                    response = self.model.generate_content([
                        formatted_template,
                        img
                    ])
                    
                    # Wait for completion and handle response
                    status.update("[bold yellow]⏳ Processing response...[/]")
                    response.resolve()
                    console.print("[green]✓[/] Received response from Gemini")
                    
                    # Print the raw response for debugging
                    console.print("\n[cyan]📝 Raw Gemini Response:[/]")
                    console.print("[dim]" + "─" * 50 + "[/]")
                    console.print(response.text)
                    console.print("[dim]" + "─" * 50 + "[/]\n")
                    
                    # Cleanup temp files
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temp files: {e}")
            else:
                img = self._load_image(image_path)
                response = self.model.generate_content([CONTENT_TEMPLATE, img])
                response.resolve()
            
            if not response.candidates or not response.text:
                if self.verbose:
                    console.print("[red]✗ No valid response generated[/]")
                raise ValueError("No valid response generated")
            
            # Extract and clean the response
            text = response.text
            
            # Find the XML content
            start = text.find('<?xml')
            if start == -1:
                start = text.find('<image_analysis>')
            
            end = text.find('</image_analysis>')
            if end == -1:
                if self.verbose:
                    console.print("[red]✗ Invalid XML format in response[/]")
                raise ValueError("No valid XML content found in response")
            
            xml_content = text[start:end + len('</image_analysis>')]
            
            if self.verbose:
                console.print("\n[bold green]✓ Analysis complete![/]")
                console.print("[cyan]📝 Parsed Analysis:[/]")
                # Print the formatted XML content
                for line in xml_content.split('\n'):
                    console.print(f"  [dim]{line}[/]")
            
            # Parse XML content into structured data
            return {
                'content': xml_content.strip(),
                'metadata': {
                    'description': 'Image analysis result',
                    'format': img.format,
                    'dimensions': f"{img.width}x{img.height}"
                }
            }
            
        except Exception as e:
            if self.verbose:
                console.print(f"[red]✗ Error: {str(e)}[/]")
            logger.error(f"Error analyzing image with Gemini: {str(e)}")
            raise Exception(f"Error analyzing image with Gemini: {str(e)}")

    async def analyze_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Analyze a batch of images."""
        results = []
        for path in image_paths:
            try:
                result = await self.generate_content(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
                if self.verbose:
                    console.print(f"[red]✗ Failed to process {path}:[/] {str(e)}")
                results.append({
                    'error': str(e),
                    'path': path
                })
        return results

def get_provider(api_key: str, model: Optional[str] = None, verbose: bool = False) -> GeminiProvider:
    """Get an instance of the Gemini provider."""
    return GeminiProvider(api_key, model, verbose)