"""Configuration module for image processing."""
import os
from pathlib import Path
from typing import Optional

class Config:
    """Configuration class for image processing."""
    
    def __init__(self):
        """Initialize configuration with default values."""
        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Image processing
        self.max_dimension = int(os.getenv('MAX_DIMENSION', '1920'))
        self.compression_quality = int(os.getenv('COMPRESSION_QUALITY', '85'))
        self.compression_enabled = os.getenv('COMPRESSION_ENABLED', 'true').lower() == 'true'
        
        # Output
        self.output_directory = os.getenv('OUTPUT_DIR', 'output')
        self.save_csv = os.getenv('SAVE_CSV_OUTPUT', 'true').lower() == 'true'
        self.save_xml = os.getenv('SAVE_XML_OUTPUT', 'true').lower() == 'true'
        self.duplicate_files = os.getenv('DUPLICATE_FILES', 'false').lower() == 'true'
        
        # Batch processing
        self.default_batch_size = int(os.getenv('DEFAULT_BATCH_SIZE', '5'))
        
        # Model settings
        self.default_model = os.getenv('DEFAULT_MODEL', 'gemini-2.0-flash-exp')
        
        # Verbose output
        self.verbose_output = os.getenv('VERBOSE_OUTPUT', 'true').lower() == 'true'
        
        # EXIF writing
        self.write_exif = os.getenv('WRITE_EXIF', 'true').lower() == 'true'

    def get_output_path(self, input_path: str, output_format: str) -> Path:
        """Get the output path for a given input path and format."""
        input_path = Path(input_path)
        output_dir = Path(self.output_directory)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_name = f"{input_path.stem}_metadata.{output_format}"
        return output_dir / output_name

# Create global config instance
config = Config() 