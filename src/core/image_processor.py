from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import os
import time
import logging
import shutil
import tempfile
import re
import google.generativeai as genai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TransferSpeedColumn
from ..core.image_utils import compress_image
from ..core.llm_handler import get_provider
from ..core.config import config

class ImageProcessor:
    # Available models based on latest documentation
    AVAILABLE_MODELS = {
        '2-flash': 'gemini-2.0-flash-exp',     # Experimental next-gen features
        '1.5-flash': 'gemini-1.5-flash',       # Fast and versatile, up to 3000 images
        '1.5-flash-8b': 'gemini-1.5-flash-8b', # High volume, lower intelligence
        '1.5-pro': 'gemini-1.5-pro',           # Complex reasoning
        'pro': 'gemini-1.5-pro',               # Alias for 1.5-pro
    }

    # Prompt template for image analysis
    PROMPT = """Analyze this image and provide a detailed description with the following structure:

Description: A clear, concise description of the main subject and scene.
Keywords: Relevant keywords and tags, comma-separated
Technical Details: Image format, dimensions, and color space
Visual Elements: List of key visual elements present
Composition: Notable composition techniques used
Mood: Overall mood or atmosphere
Use Cases: Potential applications or use cases for this image

Please be specific and detailed in your analysis."""

    def __init__(self, api_key: str, model: str = None, rename_files: bool = None, prefix: str = None, batch_size: int = None, progress_callback=None):
        """Initialize the image processor."""
        self.api_key = api_key
        self.model = model or config.default_model
        self.model_name = self.model
        self.rename_files = rename_files if rename_files is not None else config.rename_files
        self.prefix = prefix or config.file_prefix
        self.provider = get_provider(api_key, model=self.model)
        
        # Set model-specific batch size limits
        default_batch_size = batch_size or config.default_batch_size
        if model in ['gemini-2.0-flash-exp', 'gemini-1.5-flash']:
            self.batch_size = min(default_batch_size, config.max_batch_size)
        else:
            self.batch_size = 1
        
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
        
        # Progress callback
        self.progress_callback = progress_callback
        
        # Initialize console
        self.console = Console()

    def create_llm_optimized_copy(self, file_path: str, max_dimension: int = None, quality: int = None) -> tuple[str, str]:
        """Creates a temporary compressed copy of an image optimized for LLM processing."""
        max_dimension = max_dimension or config.max_dimension
        quality = quality or config.compression_quality
        
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
            raise Exception(f"Failed to create LLM-optimized copy: {str(e)}")

    def analyze_image(self, image_path: str) -> str:
        """Analyze an image using the configured provider."""
        return self.provider.analyze_image(image_path)

    def analyze_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Analyze a batch of images using the configured provider.
        
        Args:
            image_paths: List of paths to images to analyze
            
        Returns:
            List of dictionaries containing analysis results for each image
        """
        if len(image_paths) > self.batch_size:
            # Split into smaller batches if needed
            results = []
            for i in range(0, len(image_paths), self.batch_size):
                batch = image_paths[i:i + self.batch_size]
                results.extend(self.provider.analyze_batch(batch))
            return results

        # Process single batch with structured output
        try:
            # Prepare batch request with structured output schema
            schema = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "description": {"type": "string"},
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "technical_details": {
                            "type": "object",
                            "properties": {
                                "format": {"type": "string"},
                                "dimensions": {"type": "string"},
                                "color_space": {"type": "string"}
                            }
                        },
                        "visual_elements": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "composition": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "mood": {"type": "string"},
                        "use_cases": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["path", "description", "keywords"]
                }
            }

            # Process batch with structured output
            return self.provider.analyze_batch(image_paths, schema=schema)
            
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            # Return error results for each image in batch
            return [{
                'path': path,
                'success': False,
                'error': str(e)
            } for path in image_paths]

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

    async def process_images(self, folder_path: str, compress: bool = None) -> List[Dict[str, Any]]:
        """Process all images in the given folder."""
        # Use configuration value if compress is not specified
        compress = compress if compress is not None else config.compression_enabled
        
        all_results = []
        try:
            # Start timing
            self.stats['start_time'] = time.time()
            batch_start_time = time.time()
            
            # Get list of image files
            image_files = self._get_image_files(folder_path)
            self.stats['total_images'] = len(image_files)
            
            if not image_files:
                logging.warning("No image files found in the specified folder")
                return []
            
            # Calculate and display time estimate
            estimated_time = self.estimate_processing_time(len(image_files), compress)
            logging.info(f"Estimated processing time: {estimated_time}")
            
            if not compress:
                logging.warning("Processing without compression will take significantly longer. Consider using --compress for faster processing.")
            
            # Create batches
            batches = [image_files[i:i + self.batch_size] for i in range(0, len(image_files), self.batch_size)]
            self.stats['total_batches'] = len(batches)
            
            # Process batches
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TransferSpeedColumn(),
            ) as progress:
                main_task = progress.add_task(
                    f"[cyan]Processing {len(image_files)} images...",
                    total=len(image_files)
                )
                
                for batch_num, batch in enumerate(batches, 1):
                    batch_start_time = time.time()
                    self.stats['current_batch'] = batch_num
                    
                    # Process batch
                    batch_results = await self._process_batch_async(
                        batch,
                        compress=compress,
                        progress=progress,
                        task=main_task
                    )
                    all_results.extend(batch_results)
                    
                    # Update statistics
                    batch_time = time.time() - batch_start_time
                    self.stats['batch_times'].append(batch_time)
                    self.stats['avg_time_per_image'] = sum(self.stats['batch_times']) / self.stats['processed']
                    
                    # Update progress callback
                    if self.progress_callback:
                        self.progress_callback({
                            'progress': batch_num / self.stats['total_batches'],
                            'current_batch': batch_num,
                            'total_batches': self.stats['total_batches'],
                            'processed': self.stats['processed'],
                            'failed': self.stats['failed'],
                            'avg_time_per_image': self.stats['avg_time_per_image'],
                            'estimated_remaining': self.estimate_remaining_time()
                        })
                    
                    # Log batch completion
                    logging.info(f"Completed batch {batch_num}/{self.stats['total_batches']}")
                    logging.info(f"Batch processing time: {batch_time:.2f}s")
                    logging.info(f"Average time per image: {self.stats['avg_time_per_image']:.2f}s")
            
            # Log final statistics
            total_time = time.time() - self.stats['start_time']
            logging.info(f"\nProcessing complete!")
            logging.info(f"Total time: {total_time:.2f}s")
            logging.info(f"Images processed: {self.stats['processed']}")
            logging.info(f"Failed: {self.stats['failed']}")
            logging.info(f"Average time per image: {self.stats['avg_time_per_image']:.2f}s")
            
            if compress:
                compression_ratio = (
                    self.stats['compression_stats']['total_original_size'] /
                    self.stats['compression_stats']['total_compressed_size']
                )
                time_saved = self.stats['compression_stats']['time_saved']
                logging.info(f"Compression ratio: {compression_ratio:.2f}x")
                logging.info(f"Estimated time saved: {time_saved:.2f}s")
            
            return all_results
            
        except Exception as e:
            logging.error(f"Error processing images: {str(e)}")
            raise

    def estimate_remaining_time(self) -> float:
        """Estimate remaining processing time based on current statistics."""
        if not self.stats['processed']:
            return 0.0
        
        remaining_images = self.stats['total_images'] - self.stats['processed']
        return remaining_images * self.stats['avg_time_per_image'] 

    async def _process_batch_async(self, batch: List[Path], compress: bool = False, progress=None, task=None) -> List[dict]:
        """Process a batch of images asynchronously."""
        results = []
        
        try:
            # Initialize model
            model = genai.GenerativeModel(model_name=self.model_name)
            
            # Prepare images for processing
            image_data_list = []
            original_sizes = []
            compressed_paths = []
            
            for image_path in batch:
                try:
                    if compress:
                        # Create compressed copy for processing
                        original_size = os.path.getsize(image_path)
                        temp_dir, compressed_path = self.create_llm_optimized_copy(
                            str(image_path),
                            max_dimension=1024,  # Optimal for most models
                            quality=85
                        )
                        compressed_size = os.path.getsize(compressed_path)
                        
                        # Update compression statistics
                        self.stats['compression_stats']['enabled'] = True
                        self.stats['compression_stats']['total_original_size'] += original_size
                        self.stats['compression_stats']['total_compressed_size'] += compressed_size
                        
                        # Read compressed image
                        with open(compressed_path, 'rb') as f:
                            image_data = f.read()
                            image_data_list.append(image_data)
                        
                        original_sizes.append(original_size)
                        compressed_paths.append((temp_dir, compressed_path))
                        
                        # Clean up immediately after reading
                        try:
                            os.remove(compressed_path)
                            os.rmdir(temp_dir)
                        except Exception as e:
                            logging.warning(f"Error cleaning up temporary files: {str(e)}")
                        
                    else:
                        # Read original image
                        with open(image_path, 'rb') as f:
                            image_data = f.read()
                            image_data_list.append(image_data)
                        compressed_paths.append((None, None))
                    
                except Exception as e:
                    logging.error(f"Error preparing image {image_path}: {str(e)}")
                    results.append({
                        'path': str(image_path),
                        'success': False,
                        'error': str(e)
                    })
                    continue
            
            if not image_data_list:
                return results
            
            # Process images in batch
            try:
                batch_start = time.time()
                
                # Prepare batch request
                content_parts = [self.PROMPT]
                for image_data in image_data_list:
                    content_parts.append({
                        'mime_type': 'image/jpeg',
                        'data': image_data
                    })
                
                # Generate content for batch
                response = model.generate_content(content_parts)
                
                if not response.candidates:
                    raise ValueError("No response from model")
                
                # Parse batch response
                batch_results = response.text.split('\n\n')  # Assuming one response per image
                
                for i, (image_path, result_text) in enumerate(zip(batch, batch_results)):
                    try:
                        # Parse individual result
                        result = {
                            'path': str(image_path),
                            'success': True
                        }
                        
                        # Extract sections
                        sections = result_text.split('\n')
                        for section in sections:
                            if section.startswith('Description:'):
                                result['description'] = section.replace('Description:', '').strip()
                            elif section.startswith('Keywords:'):
                                result['keywords'] = [k.strip() for k in section.replace('Keywords:', '').split(',')]
                            elif section.startswith('Technical Details:'):
                                result['technical_details'] = {}
                                tech_details = section.replace('Technical Details:', '').strip()
                                for detail in tech_details.split(','):
                                    if ':' in detail:
                                        key, value = detail.split(':', 1)
                                        result['technical_details'][key.strip().lower()] = value.strip()
                            elif section.startswith('Visual Elements:'):
                                result['visual_elements'] = [e.strip() for e in section.replace('Visual Elements:', '').split(',')]
                            elif section.startswith('Composition:'):
                                result['composition'] = [c.strip() for c in section.replace('Composition:', '').split(',')]
                            elif section.startswith('Mood:'):
                                result['mood'] = section.replace('Mood:', '').strip()
                            elif section.startswith('Use Cases:'):
                                result['use_cases'] = [u.strip() for u in section.replace('Use Cases:', '').split(',')]
                        
                        # Update results
                        results.append(result)
                        self.stats['processed'] += 1
                        
                    except Exception as e:
                        logging.error(f"Error processing result for {image_path}: {str(e)}")
                        results.append({
                            'path': str(image_path),
                            'success': False,
                            'error': str(e)
                        })
                        self.stats['failed'] += 1
                    
                    finally:
                        # Clean up compressed files if used
                        if compressed_paths[i][0]:
                            try:
                                shutil.rmtree(compressed_paths[i][0], ignore_errors=True)
                            except Exception as e:
                                logging.warning(f"Error cleaning up temporary files: {str(e)}")
                        
                        # Update progress
                        if progress and task:
                            progress.update(task, advance=1)
                
                # Update timing statistics
                batch_time = time.time() - batch_start
                if compress and original_sizes:
                    # Calculate compression ratio from the original sizes we collected
                    total_original = sum(original_sizes)
                    total_compressed = sum(os.path.getsize(p[1]) for p in compressed_paths if p[1] and os.path.exists(p[1]))
                    if total_compressed > 0:  # Avoid division by zero
                        compression_ratio = total_original / total_compressed
                        self.stats['compression_stats']['time_saved'] += (batch_time * (compression_ratio - 1))
                
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)}")
                # Mark all remaining images as failed
                for image_path in batch[len(results):]:
                    results.append({
                        'path': str(image_path),
                        'success': False,
                        'error': str(e)
                    })
                    self.stats['failed'] += 1
            
            finally:
                # Clean up any remaining temporary files
                for temp_dir, _ in compressed_paths:
                    if temp_dir:
                        try:
                            shutil.rmtree(temp_dir, ignore_errors=True)
                        except Exception as e:
                            logging.warning(f"Error cleaning up temporary files: {str(e)}")
        
        except Exception as e:
            logging.error(f"Fatal error in batch processing: {str(e)}")
            raise
        
        return results 

    def _get_image_files(self, folder_path: str) -> List[Path]:
        """Get list of image files from folder."""
        folder = Path(folder_path)
        image_files = []
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif']
        
        # Find all image files
        for ext in extensions:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))
        
        return sorted(image_files)  # Sort for consistent ordering 

    def _parse_batch_results(self, batch: List[Path], result_text: str) -> List[dict]:
        """Parse batch results from streaming response."""
        results = []
        
        # Split response into sections for each image
        sections = result_text.split('\n\n')
        
        for image_path, section in zip(batch, sections):
            try:
                # Parse individual result
                result = {
                    'path': str(image_path),
                    'success': True
                }
                
                # Extract sections
                lines = section.split('\n')
                for line in lines:
                    if line.startswith('Description:'):
                        result['description'] = line.replace('Description:', '').strip()
                    elif line.startswith('Keywords:'):
                        result['keywords'] = [k.strip() for k in line.replace('Keywords:', '').split(',')]
                    elif line.startswith('Technical Details:'):
                        result['technical_details'] = {}
                        tech_details = line.replace('Technical Details:', '').strip()
                        for detail in tech_details.split(','):
                            if ':' in detail:
                                key, value = detail.split(':', 1)
                                result['technical_details'][key.strip().lower()] = value.strip()
                    elif line.startswith('Visual Elements:'):
                        result['visual_elements'] = [e.strip() for e in line.replace('Visual Elements:', '').split(',')]
                    elif line.startswith('Composition:'):
                        result['composition'] = [c.strip() for c in line.replace('Composition:', '').split(',')]
                    elif line.startswith('Mood:'):
                        result['mood'] = line.replace('Mood:', '').strip()
                    elif line.startswith('Use Cases:'):
                        result['use_cases'] = [u.strip() for u in line.replace('Use Cases:', '').split(',')]
                
                results.append(result)
                
            except Exception as e:
                logging.error(f"Error parsing result for {image_path}: {str(e)}")
                results.append({
                    'path': str(image_path),
                    'success': False,
                    'error': str(e)
                })
        
        return results 