"""LLM provider module for image analysis."""
import os
import logging
import tempfile
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from PIL import Image
from rich.console import Console
from pathlib import Path
import shutil
import absl.logging
import asyncio
import time
from . import image_utils
from .config import config
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prevent absl from writing to stderr
logging.root.removeHandler(absl.logging._absl_handler)

# Rich console for pretty output
console = Console()

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
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp", verbose: bool = False):
        """Initialize the Gemini provider."""
        self.api_key = api_key
        self.model_name = model
        self.verbose = verbose
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit = 10  # requests per second
        self.rate_window = 1.0  # seconds
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        try:
            self.model = genai.GenerativeModel(model)
            if self.verbose:
                console.print(f"[green]✓[/] Initialized Gemini model: {model}")
        except Exception as e:
            if self.verbose:
                console.print(f"[red]✗ Error initializing Gemini model:[/] {str(e)}")
            raise Exception(f"Failed to initialize Gemini model: {str(e)}")
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        time_passed = current_time - self.last_request_time
        
        if time_passed < self.rate_window:
            if self.request_count >= self.rate_limit:
                wait_time = self.rate_window - time_passed
                if self.verbose:
                    console.print(f"[yellow]⚠️ Rate limit reached, waiting {wait_time:.2f}s[/]")
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.last_request_time = time.time()
        else:
            self.request_count = 0
            self.last_request_time = current_time
        
        self.request_count += 1
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image file."""
        try:
            return Image.open(image_path)
        except Exception as e:
            if self.verbose:
                console.print(f"[red]✗ Error loading image:[/] {str(e)}")
            raise Exception(f"Failed to load image: {str(e)}")

    async def process_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of images at once.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of dictionaries containing analysis results per image
        """
        if len(image_paths) > 3600:
            raise ValueError("Maximum number of images per batch is 3600")
            
        # Check rate limit
        await self._check_rate_limit()
        
        # Load and prepare all images
        images = []
        temp_dirs = []
        
        try:
            # Create a mapping of images to their indices
            image_mapping = {f"image_{i+1}": path for i, path in enumerate(image_paths)}
            
            for path in image_paths:
                # Compress and optimize image
                temp_dir, compressed_path = image_utils.create_llm_optimized_copy(
                    path,
                    max_dimension=1024,  # Optimize for Gemini
                    quality=85
                )
                temp_dirs.append(temp_dir)
                
                # Load image
                img = self._load_image(compressed_path)
                images.append(img)
            
            # Create a custom prompt that includes image identifiers
            custom_prompt = """Analyze the following images. For each image, provide:
1. Description of the content
2. Keywords/tags
3. Visual elements present
4. Mood/atmosphere
5. Potential use cases

Format the response in XML with each image's analysis in a separate <image> tag with the image's identifier.
Example:
<image_analysis>
    <image id="image_1">
        <description>...</description>
        <keywords>...</keywords>
        <visual_elements>...</visual_elements>
        <mood>...</mood>
        <use_cases>...</use_cases>
    </image>
    <image id="image_2">...</image>
</image_analysis>"""

            # Create the prompt parts and generate content
            response = self.model.generate_content([
                custom_prompt,
                *images
            ])
            
            # Wait for completion and handle response
            response.resolve()
            
            if not response.candidates or not response.text:
                raise ValueError("No valid response generated")
            
            # Extract and clean the response
            text = response.text
            
            # Find the XML content
            start = text.find('<?xml')
            if start == -1:
                start = text.find('<image_analysis>')
            
            end = text.find('</image_analysis>')
            if end == -1:
                raise ValueError("No valid XML content found in response")
            
            xml_content = text[start:end + len('</image_analysis>')]
            
            # Parse the XML to get individual image results
            results = []
            root = ET.fromstring(xml_content)
            
            for image_elem in root.findall('.//image'):
                image_id = image_elem.get('id', '')
                if image_id in image_mapping:
                    original_path = image_mapping[image_id]
                    
                    # Extract analysis for this specific image
                    description = image_elem.find('description').text if image_elem.find('description') is not None else ''
                    keywords = [k.strip() for k in image_elem.find('keywords').text.split(',')] if image_elem.find('keywords') is not None else []
                    visual_elements = [v.strip() for v in image_elem.find('visual_elements').text.split(',')] if image_elem.find('visual_elements') is not None else []
                    mood = image_elem.find('mood').text if image_elem.find('mood') is not None else ''
                    use_cases = [u.strip() for u in image_elem.find('use_cases').text.split(',')] if image_elem.find('use_cases') is not None else []
                    
                    results.append({
                        'path': original_path,
                        'content': {
                            'description': description,
                            'keywords': keywords,
                            'visual_elements': visual_elements,
                            'mood': mood,
                            'use_cases': use_cases
                        }
                    })
            
            return results
            
        except Exception as e:
            if self.verbose:
                console.print(f"[red]✗ Error processing batch: {str(e)}[/]")
            raise e
        
        finally:
            # Clean up temp files
            for temp_dir in temp_dirs:
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

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

    async def generate_content(self, image_path: str) -> Dict[str, Any]:
        """Generate content for an image using Gemini.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Check rate limit
            await self._check_rate_limit()
            
            # Compress and optimize image
            temp_dir, compressed_path = image_utils.create_llm_optimized_copy(
                image_path,
                max_dimension=1024,  # Optimize for Gemini
                quality=85
            )
            
            # Load image
            img = self._load_image(compressed_path)
            
            # Create a custom prompt that includes image identifiers
            custom_prompt = """Analyze this image and provide:
1. Description of the content
2. Keywords/tags
3. Visual elements present
4. Mood/atmosphere
5. Potential use cases

Format the response in XML with each image's analysis in a separate <image> tag with the image's identifier.
Example:
<image_analysis>
    <image id="image_1">
        <description>...</description>
        <keywords>...</keywords>
        <visual_elements>...</visual_elements>
        <mood>...</mood>
        <use_cases>...</use_cases>
    </image>
</image_analysis>"""

            # Create the prompt parts and generate content
            response = self.model.generate_content([
                custom_prompt,
                img
            ])
            
            # Wait for completion and handle response
            response.resolve()
            
            if not response.candidates or not response.text:
                raise ValueError("No valid response generated")
            
            # Extract and clean the response
            text = response.text
            
            # Find the XML content
            start = text.find('<?xml')
            if start == -1:
                start = text.find('<image_analysis>')
            
            end = text.find('</image_analysis>')
            if end == -1:
                raise ValueError("No valid XML content found in response")
            
            xml_content = text[start:end + len('</image_analysis>')]
            
            # Parse the XML to get individual image results
            results = []
            root = ET.fromstring(xml_content)
            
            for image_elem in root.findall('.//image'):
                image_id = image_elem.get('id', '')
                
                # Extract analysis for this specific image
                description = image_elem.find('description').text if image_elem.find('description') is not None else ''
                keywords = [k.strip() for k in image_elem.find('keywords').text.split(',')] if image_elem.find('keywords') is not None else []
                visual_elements = [v.strip() for v in image_elem.find('visual_elements').text.split(',')] if image_elem.find('visual_elements') is not None else []
                mood = image_elem.find('mood').text if image_elem.find('mood') is not None else ''
                use_cases = [u.strip() for u in image_elem.find('use_cases').text.split(',')] if image_elem.find('use_cases') is not None else []
                
                results.append({
                    'path': image_path,
                    'content': {
                        'description': description,
                        'keywords': keywords,
                        'visual_elements': visual_elements,
                        'mood': mood,
                        'use_cases': use_cases
                    }
                })
            
            return results[0]
            
        except Exception as e:
            if self.verbose:
                console.print(f"[red]✗ Error processing image: {str(e)}[/]")
            raise e
        
        finally:
            # Clean up temp files
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

def get_provider(api_key: str, model: str = "gemini-2.0-flash-exp", verbose: bool = False) -> GeminiProvider:
    """Get an instance of the Gemini provider."""
    return GeminiProvider(api_key, model, verbose)