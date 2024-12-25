from typing import Any, Optional
import os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """Configuration handler for Image Sense."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        # Load .env file if it exists
        env_path = Path('.env')
        load_dotenv(dotenv_path=env_path)
        
        # Image Processing Configuration
        self.compression_enabled = self._get_bool('COMPRESSION_ENABLED', True)
        self.compression_quality = self._get_int('COMPRESSION_QUALITY', 85)
        self.max_dimension = self._get_int('MAX_DIMENSION', 1024)
        
        # Batch Processing
        self.default_batch_size = self._get_int('DEFAULT_BATCH_SIZE', 8)
        self.max_batch_size = self._get_int('MAX_BATCH_SIZE', 16)
        
        # Output Configuration
        self.default_output_format = self._get_str('DEFAULT_OUTPUT_FORMAT', 'csv')
        self.output_directory = self._get_str('OUTPUT_DIRECTORY', 'output')
        
        # Model Configuration
        self.default_model = self._get_str('DEFAULT_MODEL', 'gemini-2.0-flash-exp')
        self.rename_files = self._get_bool('RENAME_FILES', False)
        self.file_prefix = self._get_str('FILE_PREFIX', '')
        
        # Metadata Configuration
        self.backup_metadata = self._get_bool('BACKUP_METADATA', True)
        self.write_exif = self._get_bool('WRITE_EXIF', True)
        
        # Progress and Logging
        self.show_progress = self._get_bool('SHOW_PROGRESS', True)
        self.log_level = self._get_str('LOG_LEVEL', 'INFO')
        
        # Create output directory if it doesn't exist
        Path(self.output_directory).mkdir(parents=True, exist_ok=True)
    
    def _get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def _get_int(self, key: str, default: int) -> int:
        """Get integer value from environment variable."""
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default
    
    def _get_str(self, key: str, default: str) -> str:
        """Get string value from environment variable."""
        return os.getenv(key, default)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return getattr(self, key.lower(), default)
    
    def update(self, **kwargs) -> None:
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self, key.lower()):
                setattr(self, key.lower(), value)
    
    @classmethod
    def load(cls) -> 'Config':
        """Load configuration from environment."""
        return cls()

# Global configuration instance
config = Config.load() 