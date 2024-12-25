"""
Settings manager for MetaMind application.
"""
import os
from typing import Dict, Any
import json
from pathlib import Path

class Settings:
    """Settings configuration class"""
    
    # Default settings
    defaults = {
        "provider": "gemini",  # gemini only
        "output_format": "csv",  # csv or xml
        "gemini": {
            "model": "gemini-pro-vision",
            "temperature": 0.7,
            "description": "Google's powerful vision model with fast processing"
        }
    }
    
    def __init__(self):
        self.settings = self.defaults.copy()
        self.load_settings()
    
    def load_settings(self):
        """Load settings from config file if it exists"""
        config_path = Path.home() / '.image_processor' / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                loaded = json.load(f)
                self.settings.update(loaded)
    
    def save_settings(self):
        """Save current settings to config file"""
        config_path = Path.home() / '.image_processor' / 'config.json'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.settings, f, indent=4)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a setting value"""
        self.settings[key] = value
        self.save_settings()
    
    def get_provider_settings(self, provider: str) -> Dict[str, Any]:
        """Get settings for a specific provider"""
        return self.settings.get(provider, {}) 