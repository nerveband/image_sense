class ImageProcessor:
    # Available models based on latest documentation
    AVAILABLE_MODELS = {
        '2-flash': 'gemini-2.0-flash-exp',     # Experimental next-gen features
        '1.5-flash': 'gemini-1.5-flash',       # Fast and versatile, up to 3000 images
        '1.5-flash-8b': 'gemini-1.5-flash-8b', # High volume, lower intelligence
        '1.5-pro': 'gemini-1.5-pro',           # Complex reasoning
        'pro': 'gemini-1.5-pro',               # Alias for 1.5-pro
        'claude-haiku': 'claude-haiku',         # Claude Haiku model, up to 100 images
    }

    def __init__(self, api_key: str, model: str = '1.5-pro', rename_files: bool = False, prefix: str = None, batch_size: int = 100, progress_callback=None):
        """Initialize the image processor."""
        self.api_key = api_key
        self.model = model
        self.rename_files = rename_files
        self.prefix = prefix
        
        # Set model-specific batch size limits
        if model in ['1.5-flash', '2-flash']:
            self.batch_size = min(batch_size, 3000)
        elif model == 'claude-haiku':
            self.batch_size = min(batch_size, 100)
        else:
            self.batch_size = min(batch_size, 50)  # Default conservative limit
        
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

    def estimate_processing_time(self, total_images: int, compressed: bool = False) -> str:
        """Estimate total processing time based on batch size and model."""
        # Base time estimates (in seconds per image)
        base_times = {
            '1.5-flash': 1.5,
            '2-flash': 2.0,
            'claude-haiku': 3.0,
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

    async def process_images(self, folder_path: str, compress: bool = False):
        """Process all images in the given folder."""
        try:
            # Start timing
            self.stats['start_time'] = time.time()
            batch_start_time = time.time()
            
            # Get list of image files
            image_files = self._get_image_files(folder_path)
            self.stats['total_images'] = len(image_files)
            
            if not image_files:
                logging.warning("No image files found in the specified folder")
                return
            
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
                        temp_dir, compressed_path = create_llm_optimized_copy(
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
                        
                        original_sizes.append(original_size)
                        compressed_paths.append((temp_dir, compressed_path))
                        
                    else:
                        # Read original image
                        with open(image_path, 'rb') as f:
                            image_data = f.read()
                        compressed_paths.append((None, None))
                    
                    image_data_list.append(image_data)
                    
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
                        description_match = re.search(r'Description: (.*?)(?:\n|$)', result_text)
                        keywords_match = re.search(r'Keywords: (.*?)(?:\n|$)', result_text)
                        
                        if not description_match or not keywords_match:
                            raise ValueError(f"Unexpected response format: {result_text}")
                        
                        description = description_match.group(1).strip()
                        keywords = [k.strip() for k in keywords_match.group(1).split(',')]
                        
                        # Write metadata
                        metadata_dict = {
                            'Description': description,
                            'Keywords': ', '.join(keywords),
                            'Software': 'AI-Powered Image Metadata Processor v2.0'
                        }
                        
                        if not self._write_metadata(str(image_path), metadata_dict):
                            raise ValueError("Failed to write metadata")
                        
                        # Update results
                        results.append({
                            'path': str(image_path),
                            'success': True,
                            'description': description,
                            'keywords': keywords
                        })
                        
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
                if compress:
                    # Estimate time saved by compression
                    uncompressed_ratio = sum(original_sizes) / sum(os.path.getsize(p[1]) for p in compressed_paths if p[1])
                    self.stats['compression_stats']['time_saved'] += (batch_time * (uncompressed_ratio - 1))
                
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