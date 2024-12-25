"""
Command-line interface for image processing
"""

import os
import sys
import click
from pathlib import Path
from typing import List

from ..core.llm_handler import get_provider
from ..core.output_handler import save_metadata
from ..core.image_utils import validate_image_path
from ..core.settings_manager import Settings

settings = Settings()

@click.group()
def cli():
    """Image Processor CLI - Generate metadata for images using AI"""
    pass

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output-format', '-f', type=click.Choice(['csv', 'xml']), default='csv',
              help='Output format for metadata (csv or xml)')
def process(image_path: str, output_format: str):
    """Process a single image and generate metadata"""
    try:
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            click.echo("Error: GOOGLE_API_KEY environment variable not set", err=True)
            sys.exit(1)

        # Validate image path
        image_path = validate_image_path(image_path)
        if not image_path:
            click.echo("Error: Invalid image path", err=True)
            sys.exit(1)

        # Get provider
        provider = get_provider(api_key)

        # Process image
        result = provider.analyze_image(image_path)

        # Save output
        output_path = save_metadata(result, image_path, output_format)
        click.echo(f"Metadata saved to: {output_path}")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--output-format', '-f', type=click.Choice(['csv', 'xml']), default='csv',
              help='Output format for metadata (csv or xml)')
@click.option('--recursive', '-r', is_flag=True, help='Process directories recursively')
def bulk_process(directory: str, output_format: str, recursive: bool):
    """Process all images in a directory"""
    try:
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            click.echo("Error: GOOGLE_API_KEY environment variable not set", err=True)
            sys.exit(1)

        # Get provider
        provider = get_provider(api_key)

        # Get image paths
        directory_path = Path(directory)
        pattern = '**/*' if recursive else '*'
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            image_paths.extend(directory_path.glob(f'{pattern}{ext}'))

        if not image_paths:
            click.echo("No images found in directory", err=True)
            sys.exit(1)

        # Process each image
        with click.progressbar(image_paths, label='Processing images') as bar:
            for image_path in bar:
                try:
                    result = provider.analyze_image(str(image_path))
                    output_path = save_metadata(result, str(image_path), output_format)
                    click.echo(f"Processed: {image_path.name} -> {output_path}")
                except Exception as e:
                    click.echo(f"Error processing {image_path.name}: {str(e)}", err=True)

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    cli() 