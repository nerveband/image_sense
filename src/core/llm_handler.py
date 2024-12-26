"""LLM provider module for image analysis."""
import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from PIL import Image

from .config import config

CONTENT_TEMPLATE = """Analyze this image and provide a detailed description in XML format.
Please structure your response as a valid XML document with the following elements:

<?xml version="1.0" encoding="UTF-8"?>
<image_analysis>
    <description>A clear, concise description of the main subject and scene</description>
    <keywords>
        <keyword>key term 1</keyword>
        <keyword>key term 2</keyword>
        <!-- Add more keywords as needed -->
    </keywords>
    <technical_details>Technical aspects of the image including format, dimensions, and color space</technical_details>
    <visual_elements>Notable visual elements and their arrangement in the image</visual_elements>
    <composition>Description of composition techniques used</composition>
    <mood>Overall mood or emotional impact of the image</mood>
    <use_cases>Potential applications or use cases for the image</use_cases>
    <suggested_filename>descriptive_filename.jpg</suggested_filename>
</image_analysis>

Important:
1. Return ONLY the XML document - no additional text before or after
2. Ensure all XML elements are properly closed
3. Use descriptive text for each element
4. For keywords, include at least 3-5 relevant terms
5. Make the suggested filename descriptive and use underscores between words
6. Keep all text content clear and concise"""

class GeminiProvider:
    """Provider class for Gemini API."""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini provider."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=config.default_model,
            generation_config={
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 2048,
            },
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        )

    async def analyze_image(self, image_path: str) -> str:
        """Analyze a single image using Gemini."""
        try:
            # Load and prepare the image
            img = Image.open(image_path)
            
            # Generate content
            response = self.model.generate_content([CONTENT_TEMPLATE, img])
            
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
            return xml_content.strip()
            
        except Exception as e:
            raise Exception(f"Error analyzing image with Gemini: {str(e)}")

    async def analyze_batch(self, image_paths: List[str]) -> List[str]:
        """Analyze a batch of images using Gemini."""
        results = []
        for image_path in image_paths:
            try:
                result = await self.analyze_image(image_path)
                results.append(result)
            except Exception as e:
                results.append(str(e))
        return results

def get_provider(api_key: str) -> GeminiProvider:
    """Get the LLM provider instance."""
    return GeminiProvider(api_key) 