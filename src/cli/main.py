"""
Command-line interface for image processing
"""

import os
import sys
import click
import asyncio
from pathlib import Path
from typing import Optional
import logging
import pandas as pd
import xml.etree.ElementTree as ET
import shutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TransferSpeedColumn
import google.generativeai as genai
from dotenv import load_dotenv

from ..core.image_processor import ImageProcessor
from ..core.metadata_handler import MetadataHandler
from ..core.output_handler import OutputHandler
from ..core.config import Config

# Load environment variables
load_dotenv()

# Available models
AVAILABLE_MODELS = {
    '2-flash': 'gemini-2.0-flash-exp',     # Experimental next-gen features
    '1.5-flash': 'gemini-1.5-flash',       # Fast and versatile
    '1.5-pro': 'gemini-1.5-pro',           # Complex reasoning
    'pro': 'gemini-1.5-pro',               # Alias for 1.5-pro
}

def get_model_from_env() -> str:
    """Get model name from environment variable or use default."""
    return os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')

@click.group()
def cli():
    """Image Sense CLI for processing and analyzing images."""
    # Set up logging based on configuration
    config = Config()
    logging.basicConfig(level=getattr(logging, config.log_level.upper()))

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output-format', '-f', type=click.Choice(['csv', 'xml']), default='csv',
              help='Output format for metadata (csv or xml)')
@click.option('--no-compress', is_flag=True, help='Disable image compression (not recommended for large files)')
@click.option('--model', type=click.Choice(list(AVAILABLE_MODELS.keys())), default=None,
              help='Model to use for processing')
def process(image_path: str, output_format: str, no_compress: bool, model: Optional[str] = None):
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

        # Initialize processor with model if specified
        processor = ImageProcessor(
            api_key=api_key,
            model=AVAILABLE_MODELS.get(model) if model else None
        )

        # Process image with compression by default
        result = asyncio.run(processor.process_single(image_path, compress=not no_compress))

        # Save output
        output_handler = OutputHandler()
        if output_format == 'csv':
            output_path = output_handler.save_to_csv([result], f"{image_path}_metadata.csv")
        else:
            output_path = output_handler.save_to_xml([result], f"{image_path}_metadata.xml")
            
        click.echo(f"Metadata saved to: {output_path}")
        sys.exit(0)

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(2)

@cli.command(name='bulk-process')
@click.argument('directory', type=click.Path(exists=True, path_type=Path))
@click.option('--api-key', required=False, help='API key for the image processing service')
@click.option('--model', type=click.Choice(list(AVAILABLE_MODELS.keys())), default=None,
              help='Model to use for processing')
@click.option('--batch-size', type=int, default=None, help='Number of images to process in parallel')
@click.option('--no-compress', is_flag=True, help='Disable image compression')
@click.option('--output-format', type=click.Choice(['csv', 'xml']), default=None, help='Output format for results')
@click.option('--output-dir', type=click.Path(path_type=Path), default=None, help='Directory to save output files')
@click.option('--rename-files', is_flag=True, default=None, help='Rename processed files')
@click.option('--prefix', default=None, help='Prefix for renamed files')
@click.option('--recursive', is_flag=True, help='Process subdirectories recursively')
def bulk_process_cmd(directory, api_key, model, batch_size, no_compress, output_format, output_dir, rename_files, prefix, recursive):
    """Process multiple images in a directory"""
    # Convert directory to absolute path
    directory = Path(directory).resolve()
    if not directory.exists():
        raise click.BadParameter(f"Directory {directory} does not exist")
        
    asyncio.run(_bulk_process(str(directory), api_key, model, batch_size, no_compress, output_format, output_dir, rename_files, prefix, recursive))

async def _bulk_process(directory, api_key, model, batch_size, no_compress, output_format, output_dir, rename_files, prefix, recursive):
    """Async implementation of bulk processing"""
    try:
        # Get API key from argument or environment
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            click.echo("Warning: No API key found in environment variables", err=True)
            api_key = click.prompt("Please enter your Google API key", type=str, hide_input=True)
            
        # Initialize processor with configuration values as defaults
        processor = ImageProcessor(
            api_key=api_key,
            model=AVAILABLE_MODELS.get(model) if model else None,
            rename_files=rename_files,
            prefix=prefix,
            batch_size=batch_size
        )
        
        # Get configuration
        config = Config()
        
        # Use configuration values if options not specified
        output_format = output_format or config.default_output_format
        output_dir = output_dir or config.output_directory
        compress = not no_compress if no_compress is not None else config.compression_enabled
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get all images in directory
        if recursive:
            click.echo("Processing directory recursively...")
            image_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        image_files.append(os.path.join(root, file))
        else:
            image_files = [
                os.path.join(directory, f) for f in os.listdir(directory)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
            ]

        if not image_files:
            click.echo("No images found in directory")
            return

        # Process images
        results = await processor.process_images(
            image_files,
            output_dir=output_dir,
            compress=compress,
            verbose=True
        )

        # Save results
        output_handler = OutputHandler()
        if output_format == 'csv':
            output_path = output_handler.save_to_csv(results, os.path.join(output_dir, 'results.csv'))
        else:
            output_path = output_handler.save_to_xml(results, os.path.join(output_dir, 'results.xml'))
            
        click.echo(f"\nResults saved to: {output_path}")
        
        # Print statistics
        success_count = sum(1 for r in results if r.get('success', False))
        fail_count = len(results) - success_count
        
        click.echo(f"\nProcessing complete!")
        click.echo(f"Successfully processed: {success_count} images")
        if fail_count > 0:
            click.echo(f"Failed to process: {fail_count} images", err=True)
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(2)

if __name__ == '__main__':
    cli() 