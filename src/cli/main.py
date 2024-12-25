"""
Command-line interface for image processing
"""

import os
import sys
import click
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from src.core.llm_handler import (
    LLMHandler,
    LLMProvider,
    LLMError,
    LLMRateLimitError,
    LLMAuthenticationError,
)
from src.core.output_handler import OutputHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_available_models(provider: str = None) -> str:
    """Get formatted string of available models"""
    models = LLMHandler.get_available_models(provider)
    if provider:
        return "\n".join([f"  - {name}: {info['description']}" if 'description' in info else f"  - {name}"
                         for name, info in models.items()])
    else:
        result = []
        for provider_name, provider_models in models.items():
            result.append(f"{provider_name.upper()} Models:")
            result.extend([f"  - {name}: {info['description']}" if 'description' in info else f"  - {name}"
                          for name, info in provider_models.items()])
            result.append("")
        return "\n".join(result)

@click.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('--provider', type=click.Choice(['gemini', 'claude']), default='gemini', help='AI provider to use')
@click.option('--model', help='Model to use (if not specified, uses provider\'s default)')
@click.option('--api-key', help='API key (if not provided, reads from environment variables)')
@click.option('--compress', is_flag=True, help='Enable image compression')
@click.option('--recursive', is_flag=True, help='Process directories recursively')
@click.option('--output', type=click.Choice(['csv', 'xml']), default='csv', help='Output format')
@click.option('--output-file', type=click.Path(), help='Output file path')
async def process_images(input: str, provider: str, model: str, api_key: str, compress: bool, recursive: bool, output: str, output_file: str) -> List[Dict[str, Any]]:
    """
    Process images based on command line arguments
    
    Args:
        input: Path to image file or directory
        provider: AI provider to use
        model: Model to use
        api_key: API key
        compress: Whether to compress images
        recursive: Whether to process directories recursively
        output: Output format (csv or xml)
        output_file: Output file path
        
    Returns:
        List[dict]: List of results for each processed image
    """
    try:
        # Get API key from environment or argument
        if not api_key:
            if provider == "gemini":
                api_key = os.getenv("GOOGLE_API_KEY")
            else:
                api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError(f"No API key provided for {provider}. Set it in environment or use --api-key")
        
        # Initialize handler
        handler = LLMHandler(
            provider=provider,
            model=model,
            api_key=api_key,
            compress_images=compress
        )
        
        # Get image paths
        if os.path.isfile(input):
            image_paths = [input]
        else:
            if not os.path.isdir(input):
                raise ValueError(f"Input path is not a file or directory: {input}")
            
            # Collect image files
            image_paths = []
            if recursive:
                for root, _, files in os.walk(input):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_paths.append(os.path.join(root, file))
            else:
                for ext in ['.jpg', '.jpeg', '.png']:
                    image_paths.extend([str(p) for p in Path(input).glob(f"*{ext}")])
                    image_paths.extend([str(p) for p in Path(input).glob(f"*{ext.upper()}")])
        
        # Process images
        results = await handler.process_batch(image_paths)
        
        # Print results
        print("\nProcessing Results:")
        print("-" * 50)
        
        success_count = 0
        successful_results = []
        for result in results:
            path = Path(result['path']).name
            if result['success']:
                success_count += 1
                print(f"\n✅ {path}:")
                metadata = result['metadata']
                print(f"Description: {metadata['description'][:100]}...")
                print(f"Keywords: {', '.join(metadata['keywords'])}")
                print(f"Categories: {', '.join(metadata['categories'])}")
                successful_results.append({
                    'file': path,
                    'description': metadata['description'],
                    'keywords': metadata['keywords'],
                    'categories': metadata['categories']
                })
            else:
                print(f"\n❌ {path}:")
                print(f"Error: {result['error']}")
        
        print("\nSummary:")
        print(f"Total images: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(results) - success_count}")
        
        # Get processing stats
        stats = handler.get_processing_stats()
        if stats:
            print("\nProcessing Stats:")
            print(f"Average time per image: {stats.get('avg_time_per_image', 0):.2f}s")
            if stats.get('compression_savings'):
                print(f"Time saved with compression: {stats['compression_savings']:.2f}s per image")
        
        # Save results to file if specified
        if output_file and successful_results:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            output_handler = OutputHandler(output_dir)
            if output == 'csv':
                output_handler.export_to_csv(successful_results, os.path.basename(output_file))
                print(f"\nResults saved to {output_file}")
            else:  # xml
                output_handler.export_to_xml(successful_results, os.path.basename(output_file))
                print(f"\nResults saved to {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        raise

def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()
    
    # Run processing
    import asyncio
    asyncio.run(process_images())

if __name__ == "__main__":
    main() 