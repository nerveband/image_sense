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
from ..core.ascii_art import BANNER

# Initialize Rich console
console = Console()

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
    """Image Sense CLI for processing and analyzing images.
    
    Configuration Priority:
    1. Command line arguments (highest priority)
    2. Environment variables from .env file
    3. Default values in code (lowest priority)
    
    Example:
    $ image_sense process image.jpg --model 2-flash  # Command line args override all
    $ image_sense process image.jpg  # Uses .env settings or defaults
    """
    # Display banner
    console.print(BANNER)
    
    # Set up logging based on configuration
    config = Config()
    logging.basicConfig(level=getattr(logging, config.log_level.upper()))

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output-format', '-f', type=click.Choice(['csv', 'xml']), default=None,
              help='Output format for metadata (csv or xml). Overrides DEFAULT_OUTPUT_FORMAT from .env')
@click.option('--no-compress', is_flag=True, help='Disable image compression. Overrides COMPRESSION_ENABLED from .env')
@click.option('--model', type=click.Choice(list(AVAILABLE_MODELS.keys())), default=None,
              help='Model to use for processing. Overrides DEFAULT_MODEL from .env')
@click.option('--verboseoff', is_flag=True, help='Disable verbose output. Overrides VERBOSE_OUTPUT from .env')
@click.option('--api-key', help='Google API key. Overrides GOOGLE_API_KEY from .env')
@click.option('--batch-size', type=int, help='Batch size for processing. Overrides DEFAULT_BATCH_SIZE from .env')
@click.option('--output-dir', help='Output directory. Overrides OUTPUT_DIRECTORY from .env')
@click.option('--auto-append', is_flag=True, help='Automatically append to existing CSV files. Overrides AUTO_APPEND_CSV from .env')
def process(image_path: str, output_format: str, no_compress: bool, model: Optional[str] = None, 
           verboseoff: bool = False, api_key: Optional[str] = None, batch_size: Optional[int] = None,
           output_dir: Optional[str] = None, auto_append: Optional[bool] = None):
    """Process a single image.
    
    All command line options override their corresponding environment variables from .env.
    If an option is not specified, the value from .env will be used.
    If neither is specified, built-in defaults will be used.
    """
    try:
        # Get API key with priority order
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            click.echo("Error: GOOGLE_API_KEY not provided via --api-key or in .env", err=True)
            sys.exit(1)

        # Get verbose setting with priority order
        verbose = not verboseoff

        # Get output format with priority order
        output_format = output_format or os.getenv('DEFAULT_OUTPUT_FORMAT', 'csv')

        # Get auto append setting with priority order
        should_append = auto_append if auto_append is not None else os.getenv('AUTO_APPEND_CSV', 'false').lower() == 'true'

        # Initialize processor with model if specified
        processor = ImageProcessor(
            api_key=api_key,
            model=AVAILABLE_MODELS.get(model) if model else get_model_from_env(),
            verbose_output=verbose,
            batch_size=batch_size
        )

        # Process image
        result = asyncio.run(processor.analyze_image(image_path))
        
        # Save output
        output_base = os.path.join(output_dir or os.getenv('OUTPUT_DIRECTORY', '.'),
                                 os.path.splitext(os.path.basename(image_path))[0])
        if output_format == 'csv':
            output_path = f"{output_base}_metadata.csv"
            # Check if file exists and we should append
            if os.path.exists(output_path) and should_append:
                if click.confirm(f"CSV file {output_path} exists. Do you want to append to it?", default=True):
                    processor.save_to_csv([result], output_path, append=True)
                else:
                    processor.save_to_csv([result], output_path, append=False)
            else:
                processor.save_to_csv([result], output_path, append=False)
        else:
            output_path = f"{output_base}_metadata.xml"
            processor.save_to_xml([result], output_path)
            
        click.echo(f"Processing complete. Metadata saved to: {output_path}")
        return 0

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1

@cli.command('bulk-process')
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output-format', '-f', type=click.Choice(['csv', 'xml']), default=None,
              help='Output format for results. Overrides DEFAULT_OUTPUT_FORMAT from .env')
@click.option('--recursive', '-r', is_flag=True, help='Process subdirectories recursively')
@click.option('--compress/--no-compress', default=None, help='Enable/disable compression. Overrides COMPRESSION_ENABLED from .env')
@click.option('--model', '-m', type=click.Choice(['2-flash', '1.5-flash', '1.5-pro', 'pro']),
              help='Model to use for processing. Overrides DEFAULT_MODEL from .env')
@click.option('--verboseoff', is_flag=True, help='Disable verbose output. Overrides VERBOSE_OUTPUT from .env')
@click.option('--api-key', help='Google API key. Overrides GOOGLE_API_KEY from .env')
@click.option('--batch-size', type=int, help='Batch size for processing. Overrides DEFAULT_BATCH_SIZE from .env')
@click.option('--output-dir', help='Output directory. Overrides OUTPUT_DIRECTORY from .env')
@click.option('--auto-append', is_flag=True, help='Automatically append to existing CSV files. Overrides AUTO_APPEND_CSV from .env')
def bulk_process(input_path: str, output_format: str, recursive: bool, compress: bool, 
                model: Optional[str] = None, verboseoff: bool = False, api_key: Optional[str] = None,
                batch_size: Optional[int] = None, output_dir: Optional[str] = None, auto_append: Optional[bool] = None):
    """Process multiple images in a directory.
    
    All command line options override their corresponding environment variables from .env.
    If an option is not specified, the value from .env will be used.
    If neither is specified, built-in defaults will be used.
    """
    try:
        # Get API key with priority order
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            click.echo("Error: GOOGLE_API_KEY not provided via --api-key or in .env", err=True)
            sys.exit(1)

        # Get verbose setting with priority order
        verbose = not verboseoff

        # Get output format with priority order
        output_format = output_format or os.getenv('DEFAULT_OUTPUT_FORMAT', 'csv')

        # Get auto append setting with priority order
        should_append = auto_append if auto_append is not None else os.getenv('AUTO_APPEND_CSV', 'false').lower() == 'true'

        # Initialize processor with model if specified
        processor = ImageProcessor(
            api_key=api_key,
            model=AVAILABLE_MODELS.get(model) if model else get_model_from_env(),
            verbose_output=verbose,
            batch_size=batch_size
        )

        # Get all image files
        if recursive:
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif']:
                image_files.extend(Path(input_path).rglob(f"*{ext}"))
        else:
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif']:
                image_files.extend(Path(input_path).glob(f"*{ext}"))

        if not image_files:
            click.echo("No image files found in the specified directory", err=True)
            return 1

        # Process images
        results = asyncio.run(processor.process_batch(image_files))
        
        # Save results
        output_dir = output_dir or os.getenv('OUTPUT_DIRECTORY', os.path.dirname(input_path))
        if output_format == 'csv':
            output_path = os.path.join(output_dir, 'results.csv')
            # Check if file exists and we should append
            if os.path.exists(output_path) and should_append:
                if click.confirm(f"CSV file {output_path} exists. Do you want to append to it?", default=True):
                    processor.save_to_csv(results, output_path, append=True)
                else:
                    processor.save_to_csv(results, output_path, append=False)
            else:
                processor.save_to_csv(results, output_path, append=False)
        else:
            output_path = os.path.join(output_dir, 'results.xml')
            processor.save_to_xml(results, output_path)

        click.echo(f"Results saved to {output_path}")
        return 0

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1

@cli.command()
def version():
    """Show version information"""
    try:
        click.echo("Image Sense v0.1.1")
        return 0
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1

if __name__ == '__main__':
    cli()