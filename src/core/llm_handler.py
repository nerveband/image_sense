"""
LLM Handler for processing images using multiple vision model providers
"""

import base64
import os
from typing import List

from google.generativeai import GenerativeModel
import google.generativeai as genai

class GeminiProvider:
    """Google's Gemini Vision provider"""
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = GenerativeModel('gemini-pro-vision')

    def analyze_image(self, image_path: str) -> str:
        """Analyze image using Gemini Vision API"""
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
            
            prompt = """Analyze this image and provide detailed metadata including:
            1. Description of the main subject
            2. Key visual elements
            3. Colors and composition
            4. Mood or atmosphere
            5. Technical aspects (if apparent)
            6. Any text visible in the image
            7. Potential use cases or contexts
            Format as clear, descriptive sentences."""
            
            response = self.model.generate_content([prompt, image_parts[0]])
            return response.text

def get_provider(api_key: str, provider: str = "gemini") -> GeminiProvider:
    """Get the appropriate provider based on settings"""
    if provider == "gemini":
        return GeminiProvider(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}") 