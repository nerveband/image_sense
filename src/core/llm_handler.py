"""
LLM Handler for processing images using multiple vision model providers
"""

import os
import base64
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod

import google.generativeai as genai
from anthropic import Anthropic
from PIL import Image
import anthropic

class LLMError(Exception):
    """Base class for LLM-related errors"""
    pass

class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded"""
    pass

class LLMAuthenticationError(LLMError):
    """Raised when authentication fails"""
    pass

class LLMProvider(str, Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"
    CLAUDE = "claude"

# Available models
GEMINI_MODELS = ["gemini-2.0-flash-exp", "gemini-1.5-flash"]
CLAUDE_MODELS = ["claude-3-haiku-20240307", "claude-3-sonnet-20240229"]

class BaseProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def process_image(self, image: Image.Image, prompt: str) -> str:
        """Process an image with a prompt"""
        pass

    @abstractmethod
    def validate_api_key(self) -> bool:
        """Validate the API key"""
        pass

class GeminiProvider(BaseProvider):
    """Google's Gemini Vision provider"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp", temperature: float = 0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)
    
    def process_image(self, image: Image.Image, prompt: str) -> str:
        response = self.model_instance.generate_content(
            [prompt, image],
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature
            )
        )
        return response.text.strip()
    
    def validate_api_key(self) -> bool:
        try:
            response = self.model_instance.generate_content("Test")
            return True
        except Exception:
            return False

class ClaudeProvider(BaseProvider):
    """Anthropic's Claude Vision provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307", temperature: float = 0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def process_image(self, image: Image.Image, prompt: str) -> str:
        # Convert image to base64
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Create message with image
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=self.temperature,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64
                        }
                    }
                ]
            }]
        )
        
        # Extract just the comma-separated keywords if this is a keywords prompt
        response_text = message.content[0].text
        if "keywords" in prompt.lower():
            # Find the last comma-separated list in the text
            parts = response_text.split('\n')
            for part in reversed(parts):
                if ',' in part:
                    return part.strip()
            # If no comma-separated list found, return the whole response
            return response_text
        
        return response_text
    
    def validate_api_key(self) -> bool:
        try:
            # Simple test message without image
            message = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": "Test"
                }]
            )
            return True
        except Exception:
            return False

class LLMHandler:
    """Handles interactions with various vision model providers"""
    
    # Available models
    AVAILABLE_MODELS = {
        'gemini': {
            'gemini-2.0-flash-exp': {
                'description': 'Experimental next-gen features'
            },
            'gemini-1.5-flash': {
                'description': 'Fast and versatile, up to 3000 images'
            }
        },
        'claude': {
            'claude-3-haiku-20240307': {
                'description': 'Latest Claude Haiku model'
            },
            'claude-3-sonnet-20240229': {
                'description': 'Latest Claude Sonnet model'
            }
        }
    }
    
    def __init__(self, api_key: str, provider: LLMProvider = LLMProvider.GEMINI, model: str = "gemini-2.0-flash-exp", temperature: float = 0.7, compress_images: bool = False):
        """
        Initialize the LLM handler
        
        Args:
            api_key: Provider API key
            provider: Provider name (LLMProvider.GEMINI or LLMProvider.CLAUDE)
            model: Model name to use
            temperature: Temperature for generation (0.0 to 1.0)
            compress_images: Whether to compress images before processing
        """
        self.api_key = api_key
        self.provider_name = provider
        self.model = model
        self.temperature = temperature
        self.compress_images = compress_images
        
        # Initialize provider
        self.provider = self._create_provider()
    
    def _create_provider(self) -> BaseProvider:
        """Create the appropriate provider instance"""
        if self.provider_name == LLMProvider.GEMINI:
            return GeminiProvider(self.api_key, self.model, self.temperature)
        elif self.provider_name == LLMProvider.CLAUDE:
            return ClaudeProvider(self.api_key, self.model, self.temperature)
        else:
            raise ValueError(f"Unsupported provider: {self.provider_name}")
    
    async def process_image(self, image_path: str, prompts: Dict[str, str]) -> Dict[str, Any]:
        """
        Process an image using the selected provider
        
        Args:
            image_path: Path to the image file
            prompts: Dictionary of prompts for different metadata types
                    (e.g., {'description': '...', 'keywords': '...'})
        
        Returns:
            Dictionary containing the generated metadata
        """
        try:
            # Load and prepare the image
            img = Image.open(image_path)
            
            # Process each prompt type
            results = {}
            for metadata_type, prompt in prompts.items():
                results[metadata_type] = self.provider.process_image(img, prompt)
            
            return {
                'success': True,
                'metadata': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_batch(self, image_paths: List[str], prompts: Dict[str, str], 
                     callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of images
        
        Args:
            image_paths: List of paths to image files
            prompts: Dictionary of prompts for different metadata types
            callback: Optional callback function for progress updates
        
        Returns:
            List of dictionaries containing the generated metadata for each image
        """
        results = []
        total = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            result = self.process_image(image_path, prompts)
            results.append(result)
            
            if callback:
                progress = ((i + 1) / total) * 100
                callback(progress, image_path)
        
        return results
    
    @staticmethod
    def get_default_prompts() -> Dict[str, str]:
        """Get default prompts for different metadata types"""
        return {
            'description': 'Describe this image in detail, focusing on its main subjects, composition, and notable features.',
            'keywords': 'Respond ONLY with a comma-separated list of 5-10 keywords that describe what you can actually see in this image. Do not hallucinate or imagine content that is not visibly present. If the image is blank or empty, use appropriate keywords like "blank", "empty", "white", etc.',
            'categories': 'Suggest appropriate categories or tags for organizing this image, considering both general and specific classifications.'
        }
    
    def validate_api_key(self) -> bool:
        """
        Validate the API key by making a test request
        
        Returns:
            bool: True if the API key is valid, False otherwise
        """
        return self.provider.validate_api_key() 
    
    @staticmethod
    def get_available_models(provider: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Get available models for a provider or all providers
        
        Args:
            provider: Provider name (optional)
            
        Returns:
            Dictionary of available models with their descriptions
        """
        if provider:
            return LLMHandler.AVAILABLE_MODELS.get(provider, {})
        return LLMHandler.AVAILABLE_MODELS 
    
    @staticmethod
    def get_default_model(provider: str) -> str:
        """Get default model for a provider"""
        if provider == LLMProvider.GEMINI:
            return GEMINI_MODELS[0]
        elif provider == LLMProvider.CLAUDE:
            return CLAUDE_MODELS[0]
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _compress_image(self, image: Image.Image) -> Image.Image:
        """Compress image while maintaining quality"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Calculate target size (maintain aspect ratio)
        max_size = 1024
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        
        # Resize and compress
        compressed = image.resize(new_size, Image.Resampling.LANCZOS)
        return compressed

    async def process_batch(self, image_paths: List[str], prompts: Optional[Dict[str, str]] = None, 
                          callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """Process a batch of images"""
        if prompts is None:
            prompts = self.get_default_prompts()
            
        results = []
        total = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            try:
                result = await self.process_image(image_path, prompts)
                results.append(result)
                if callback:
                    progress = ((i + 1) / total) * 100
                    callback(progress, image_path)
            except Exception as e:
                results.append({
                    'file': image_path,
                    'error': str(e)
                })
        return results 