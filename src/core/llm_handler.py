"""
LLM Handler for processing images using multiple vision model providers
"""

import base64
import os
from typing import List, Dict, Any

from google.generativeai import GenerativeModel
import google.generativeai as genai

class GeminiProvider:
    """Google's Gemini Vision provider"""
    def __init__(self, api_key: str, model: str = 'gemini-2.0-flash-exp'):
        genai.configure(api_key=api_key)
        self.model = GenerativeModel(model)
        self.prompt = """Analyze this image and provide detailed metadata including:
            1. Description of the main subject
            2. Key visual elements
            3. Colors and composition
            4. Mood or atmosphere
            5. Technical aspects (if apparent)
            6. Any text visible in the image
            7. Potential use cases or contexts
            Format as clear, descriptive sentences."""

    def analyze_image(self, image_path: str) -> str:
        """Analyze a single image using Gemini Vision API"""
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
            response = self.model.generate_content([self.prompt, image_parts[0]])
            return response.text

    def analyze_batch(self, image_paths: List[str]) -> List[str]:
        """Analyze multiple images in a single API call"""
        if len(image_paths) == 1:
            return [self.analyze_image(image_paths[0])]

        # Prepare batch request
        content = [self.prompt]
        for image_path in image_paths:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                content.append({"mime_type": "image/jpeg", "data": image_data})

        # Generate content for batch
        response = self.model.generate_content(content)
        
        # Split response into individual image results
        # The response format depends on the model's output
        if hasattr(response, 'text'):
            # Split by double newline as each image analysis is separated by a blank line
            results = response.text.split('\n\n')
            # Ensure we have a result for each image
            if len(results) != len(image_paths):
                # If mismatch, process each image individually as fallback
                return [self.analyze_image(path) for path in image_paths]
            return results
        else:
            # Fallback to individual processing if batch fails
            return [self.analyze_image(path) for path in image_paths]

def get_provider(api_key: str, provider: str = "gemini", model: str = None) -> GeminiProvider:
    """Get the appropriate provider based on settings"""
    if provider == "gemini":
        return GeminiProvider(api_key, model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}") 