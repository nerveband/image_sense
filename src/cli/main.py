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
from rich.table import Table
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TransferSpeedColumn
import google.generativeai as genai
from dotenv import load_dotenv
from ..core.config import config

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
    """Image Sense CLI for processing and analyzing images."""
    # Display banner
    console.print(BANNER)
    
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
@click.option('--verboseoff', is_flag=True, help='Disable verbose output (verbose is on by default)')
def process(image_path: str, output_format: str, no_compress: bool, model: Optional[str] = None, verboseoff: bool = False):
    """Process a single image"""
    try:
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            click.echo("Error: GOOGLE_API_KEY environment variable not set", err=True)
            sys.exit(1)

        # Get verbose setting from environment or use default (True)
        verbose = not verboseoff

        # Initialize processor with model if specified
        processor = ImageProcessor(
            api_key=api_key,
            model=AVAILABLE_MODELS.get(model) if model else get_model_from_env(),
            verbose_output=verbose
        )

        # Process image with compression by default
        result = asyncio.run(processor.analyze_image(image_path))
        
        # Save output
        output_base = os.path.splitext(image_path)[0]
        if output_format == 'csv':
            output_path = f"{output_base}_metadata.csv"
            processor.save_to_csv([result], output_path)
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
@click.option('--output-format', '-f', type=click.Choice(['csv', 'xml']), default='csv',
              help='Output format for results')
@click.option('--recursive', '-r', is_flag=True, help='Process subdirectories recursively')
@click.option('--compress', '-c', is_flag=True, help='Compress images before processing')
@click.option('--model', '-m', type=click.Choice(['2-flash', '1.5-flash', '1.5-pro', 'pro']),
              help='Model to use for processing')
@click.option('--verboseoff', is_flag=True, help='Disable verbose output (verbose is on by default)')
def bulk_process(input_path: str, output_format: str, recursive: bool, compress: bool, 
                model: Optional[str] = None, verboseoff: bool = False):
    """Process multiple images in a directory."""
    try:
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            click.echo("Error: GOOGLE_API_KEY environment variable not set", err=True)
            sys.exit(1)
            
        # Get verbose setting from environment or use default (True)
        verbose = not verboseoff and os.getenv("VERBOSE_OUTPUT", "true").lower() != "false"

        # Initialize processor
        processor = ImageProcessor(
            api_key=api_key,
            model=AVAILABLE_MODELS.get(model) if model else get_model_from_env(),
            verbose_output=verbose
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
        
        # Get output directory and base filename from input path
        output_dir = input_path
        folder_name = os.path.basename(os.path.normpath(input_path))
        
        # Always save CSV by default
        csv_path = os.path.join(output_dir, f"{folder_name}_results.csv")
        processor.save_to_csv(results, csv_path)
        click.echo(f"Results saved to {csv_path}")
        
        # Pretty print results in terminal
        if verbose:
            # Create a table
            table = Table(
                title="📊 Analysis Results", 
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
                show_lines=True,
                width=None,
                pad_edge=False,
                collapse_padding=True
            )
            
            # Add columns with better formatting
            table.add_column("📁 Original Filename", style="cyan", no_wrap=True)
            table.add_column("📝 Description", style="green", max_width=60, overflow="fold")
            table.add_column("🏷️ Keywords", style="yellow", max_width=30, overflow="fold")
            table.add_column("💾 Suggested Filename", style="blue", no_wrap=True)
            
            # Add rows with better data handling
            for result in results:
                description = result.get('description', '').strip()
                keywords = result.get('keywords', [])
                if isinstance(keywords, str):
                    keywords = keywords.split(';')
                keywords_str = '\n'.join(k.strip() for k in keywords if k.strip())
                
                table.add_row(
                    result.get('original_filename', ''),
                    description,
                    keywords_str,
                    result.get('suggested_filename', '')
                )
            
            # Print the table with padding
            console.print("\n")
            console.print(table, justify="center")
            console.print("\n")
        
        # Save XML if enabled in environment or explicitly requested
        save_xml = os.getenv('SAVE_XML_OUTPUT', 'true').lower() == 'true' or output_format == 'xml'
        if save_xml:
            xml_path = os.path.join(output_dir, f"{folder_name}_results.xml")
            processor.save_to_xml(results, xml_path)
            click.echo(f"XML results saved to {xml_path}")
        
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