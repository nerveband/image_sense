"""
Settings manager for MetaMind application.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field

@dataclass
class LLMSettings:
    provider: str = "gemini"  # gemini, anthropic
    model: str = "gemini-pro-vision"  # Default model
    temperature: float = 0.7
    max_tokens: int = 1000
    description_prompt: str = "Describe this image in detail, focusing on visual elements, composition, and mood."
    keywords_prompt: str = "List relevant keywords for this image, separated by commas."
    api_key: Optional[str] = None
    fallback_provider: Optional[str] = None  # Fallback if primary provider fails
    retry_attempts: int = 3
    timeout_seconds: int = 30
    cache_responses: bool = True
    cache_duration_days: int = 30

    @staticmethod
    def get_available_models(provider: str) -> Dict[str, str]:
        """Get available models for a provider with their descriptions."""
        models = {
            "gemini": {
                "gemini-pro-vision": "Best for general use, balanced performance",
                "gemini-1.5-pro-vision": "Advanced reasoning and complex tasks",
                "gemini-1.5-vision-preview": "Preview of next-gen features",
                "gemini-1.5-vision": "Standard vision model",
                "gemini-1.0-pro-vision": "Legacy pro model",
                "gemini-1.0-vision-preview": "Legacy preview model",
            },
            "anthropic": {
                "claude-3-opus-20240229": "Most capable model, highest quality",
                "claude-3-sonnet-20240229": "Balanced performance and speed",
                "claude-3-haiku-20240229": "Fastest, most compact model",
            }
        }
        return models.get(provider, {})

    @staticmethod
    def get_provider_description(provider: str) -> str:
        """Get description for a provider."""
        descriptions = {
            "gemini": "Google's advanced vision model with strong performance and reliability",
            "anthropic": "Claude's state-of-the-art vision model with exceptional reasoning capabilities"
        }
        return descriptions.get(provider, "Unknown provider")

    @staticmethod
    def get_model_description(provider: str, model: str) -> str:
        """Get description for a specific model."""
        models = LLMSettings.get_available_models(provider)
        return models.get(model, "Unknown model")

@dataclass
class MetadataSettings:
    embed_metadata: bool = True
    create_backups: bool = True
    exif_description_tag: str = "ImageDescription"
    exif_keywords_tag: str = "Keywords"
    custom_tags: Dict[str, str] = field(default_factory=dict)
    preserve_original_date: bool = True
    backup_location: Optional[str] = None
    metadata_format: str = "exif"  # exif, xmp, iptc
    overwrite_existing: bool = False
    extract_existing_metadata: bool = True
    validate_metadata: bool = True

@dataclass
class BatchSettings:
    max_concurrent_processes: int = 4
    batch_size: int = 10
    auto_retry_failed: bool = True
    max_retries: int = 3
    retry_delay_seconds: int = 5
    priority_queue: bool = False
    error_handling: str = "skip"  # skip, stop, prompt
    notification_frequency: str = "batch"  # image, batch, completion
    preserve_order: bool = True
    skip_existing: bool = False
    recursive_processing: bool = False
    file_patterns: List[str] = field(default_factory=lambda: ["*.jpg", "*.jpeg", "*.png", "*.webp"])
    min_file_size_kb: int = 0
    max_file_size_mb: int = 100

@dataclass
class PerformanceSettings:
    image_preprocessing: bool = True
    max_image_dimension: int = 2048
    jpeg_quality: int = 85
    enable_gpu: bool = True
    cache_enabled: bool = True
    cache_size_mb: int = 1000
    memory_limit_mb: int = 2000
    temp_dir: Optional[str] = None
    cleanup_temp_files: bool = True
    compression_enabled: bool = True
    optimize_network_calls: bool = True
    debug_mode: bool = False

@dataclass
class ExportSettings:
    auto_export: bool = False
    export_format: str = "both"  # csv, xml, both
    export_directory: Optional[str] = None
    include_timestamps: bool = True
    include_file_info: bool = True
    csv_delimiter: str = ","
    xml_pretty_print: bool = True
    export_failed_items: bool = True
    group_by_date: bool = False
    group_by_folder: bool = False
    export_original_paths: bool = True
    include_processing_stats: bool = True
    export_metadata_only: bool = False
    filename_template: str = "{date}_{filename}"
    backup_exports: bool = True

@dataclass
class UISettings:
    theme: str = "system"  # light, dark, system
    font_size: str = "medium"  # small, medium, large
    show_previews: bool = True
    max_preview_size: int = 300
    confirm_overwrites: bool = True
    show_file_info: bool = True
    show_metadata_panel: bool = True
    show_processing_stats: bool = True
    enable_animations: bool = True
    compact_mode: bool = False
    keyboard_shortcuts: bool = True
    show_tooltips: bool = True
    preview_quality: str = "medium"  # low, medium, high
    auto_scroll: bool = True
    remember_last_directory: bool = True
    default_view: str = "grid"  # grid, list, details

@dataclass
class AppSettings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    metadata: MetadataSettings = field(default_factory=MetadataSettings)
    batch: BatchSettings = field(default_factory=BatchSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    ui: UISettings = field(default_factory=UISettings)

class SettingsManager:
    """Manages application settings with file persistence."""
    
    def __init__(self):
        self.settings_dir = self._get_settings_dir()
        self.settings_file = self.settings_dir / "settings.json"
        self.settings = self._load_settings()
    
    def _get_settings_dir(self) -> Path:
        """Get the platform-specific settings directory."""
        if os.name == 'nt':  # Windows
            base_dir = os.getenv('APPDATA')
        elif os.name == 'darwin':  # macOS
            base_dir = os.path.expanduser('~/Library/Application Support')
        else:  # Linux and others
            base_dir = os.path.expanduser('~/.config')
        
        settings_dir = Path(base_dir) / "metamind"
        settings_dir.mkdir(parents=True, exist_ok=True)
        return settings_dir
    
    def _load_settings(self) -> AppSettings:
        """Load settings from file or create default."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    data = json.load(f)
                    return self._dict_to_settings(data)
        except Exception as e:
            print(f"Error loading settings: {e}")
        
        return AppSettings()
    
    def _dict_to_settings(self, data: Dict[str, Any]) -> AppSettings:
        """Convert dictionary to settings object."""
        llm_data = data.get('llm', {})
        metadata_data = data.get('metadata', {})
        export_data = data.get('export', {})
        ui_data = data.get('ui', {})
        
        return AppSettings(
            llm=LLMSettings(**llm_data),
            metadata=MetadataSettings(**metadata_data),
            export=ExportSettings(**export_data),
            ui=UISettings(**ui_data)
        )
    
    def save_settings(self):
        """Save current settings to file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(asdict(self.settings), f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def update_settings(self, section: str, updates: Dict[str, Any]):
        """Update settings for a specific section."""
        if not hasattr(self.settings, section):
            raise ValueError(f"Invalid settings section: {section}")
        
        current = getattr(self.settings, section)
        for key, value in updates.items():
            if hasattr(current, key):
                setattr(current, key, value)
        
        self.save_settings()
    
    def get_settings(self) -> AppSettings:
        """Get current settings."""
        return self.settings
    
    def reset_section(self, section: str):
        """Reset a section to default values."""
        if not hasattr(self.settings, section):
            raise ValueError(f"Invalid settings section: {section}")
        
        default = AppSettings()
        setattr(self.settings, section, getattr(default, section))
        self.save_settings()
    
    def reset_all(self):
        """Reset all settings to defaults."""
        self.settings = AppSettings()
        self.save_settings() 