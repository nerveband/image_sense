"""
Command-line interface for image processing
"""

import os
import sys
import click
import asyncio
from pathlib import Path
from typing import Optional

from ..core.image_processor import ImageProcessor
from ..core.metadata_handler import MetadataHandler
from ..core.output_handler import save_metadata

# Available models
AVAILABLE_MODELS = {
    '2-flash': 'gemini-2.0-flash-exp',     # Experimental next-gen features
    '1.5-flash': 'gemini-1.5-flash',       # Fast and versatile
    '1.5-pro': 'gemini-1.5-pro',           # Complex reasoning
    'pro': 'gemini-1.5-pro',               # Alias for 1.5-pro
    'claude-haiku': 'claude-haiku',         # Claude Haiku model
}

def get_model_from_env() -> str:
    """Get model name from environment variable or use default."""
    return os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')

@click.group()
def cli():
    """Image Sense - AI-powered image metadata processor"""
    pass

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output-format', '-f', type=click.Choice(['csv', 'xml']), default='csv',
              help='Output format for metadata (csv or xml)')
@click.option('--no-compress', is_flag=True, help='Disable image compression (not recommended for large files)')
def process(image_path: str, output_format: str, no_compress: bool):
    """Process a single image"""
    try:
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            click.echo("Error: GOOGLE_API_KEY environment variable not set", err=True)
            sys.exit(1)

        # Validate image path
        if not os.path.isfile(image_path):
            click.echo("Error: Invalid image path", err=True)
            sys.exit(1)

        # Initialize processor with default model
        processor = ImageProcessor(api_key=api_key)

        # Process image with compression by default
        result = asyncio.run(processor.process_single(image_path, compress=not no_compress))

        # Save output
        output_path = save_metadata(result, image_path, output_format)
        click.echo(f"Metadata saved to: {output_path}")
        sys.exit(0)

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(2)

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--output-format', '-f', type=click.Choice(['csv', 'xml']), default='csv',
              help='Output format for metadata (csv or xml)')
@click.option('--recursive', '-r', is_flag=True, help='Process directories recursively')
@click.option('--no-compress', is_flag=True, help='Disable image compression (not recommended for large files)')
@click.option('--model', '-m', type=click.Choice(list(AVAILABLE_MODELS.keys())), 
              help='Gemini model to use (default: from env or flash-exp)')
@click.option('--batch-size', '-b', type=int, default=1, 
              help='Number of images to process in one API call (default: 1)')
@click.option('--output-dir', '-o', type=click.Path(), default='metadata',
              help='Output directory for metadata files (default: metadata)')
def bulk_process(directory: str, output_format: str, recursive: bool, no_compress: bool, model: str, batch_size: int, output_dir: str):
    """Process all images in a directory"""
    try:
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            click.echo("Error: GOOGLE_API_KEY environment variable not set", err=True)
            sys.exit(1)

        # Get model name
        model_name = AVAILABLE_MODELS[model] if model else get_model_from_env()
        click.echo(f"Using model: {model_name}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize image processor with selected model and batch size
        processor = ImageProcessor(api_key=api_key, model=model_name, batch_size=batch_size)

        # Process images with compression enabled by default
        results = asyncio.run(processor.process_images(directory, compress=not no_compress))

        # Save results
        if output_format == 'xml':
            output_path = os.path.join(output_dir, 'analysis_results.xml')
        else:
            output_path = os.path.join(output_dir, 'analysis_results.csv')

        # Save metadata using metadata handler
        handler = MetadataHandler()
        handler.process_batch(
            image_paths=[r['path'] for r in results] if results else [],
            operation='write',
            metadata=results,
            output_format=output_format,
            output_path=output_path
        )

        click.echo(f"Results saved to: {output_path}")
        sys.exit(0)

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(2)

if __name__ == '__main__':
    cli() 