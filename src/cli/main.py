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

from ..core.image_processor import ImageProcessor
from ..core.metadata_handler import MetadataHandler
from ..core.output_handler import save_metadata
from ..core.config import config

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
    """Image Sense CLI for processing and analyzing images."""
    # Set up logging based on configuration
    logging.basicConfig(level=getattr(logging, config.log_level.upper()))

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
@click.option('--api-key', required=True, help='API key for the image processing service')
@click.option('--model', default=None, help='Model to use for processing')
@click.option('--batch-size', type=int, default=None, help='Number of images to process in parallel')
@click.option('--no-compress', is_flag=True, help='Disable image compression')
@click.option('--output-format', type=click.Choice(['csv', 'xml']), default=None, help='Output format for results')
@click.option('--output-dir', type=click.Path(), default=None, help='Directory to save output files')
@click.option('--rename-files', is_flag=True, default=None, help='Rename processed files')
@click.option('--prefix', default=None, help='Prefix for renamed files')
async def bulk_process(directory, api_key, model, batch_size, no_compress, output_format, output_dir, rename_files, prefix):
    """Process all images in a directory."""
    try:
        # Initialize processor with configuration values as defaults
        processor = ImageProcessor(
            api_key=api_key,
            model=model,
            rename_files=rename_files,
            prefix=prefix,
            batch_size=batch_size
        )
        
        # Use configuration values if options not specified
        output_format = output_format or config.default_output_format
        output_dir = output_dir or config.output_directory
        compress = not no_compress if no_compress is not None else config.compression_enabled
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process images
        results = await processor.process_images(directory, compress=compress)
        
        # Save results
        if results:
            output_file = output_path / f"results.{output_format}"
            if output_format == 'csv':
                processor.save_to_csv(results, output_file)
            else:
                processor.save_to_xml(results, output_file)
            click.echo(f"Results saved to {output_file}")
        
        click.echo("Processing complete!")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--api-key', required=True, help='API key for the image processing service')
@click.option('--output-format', '-f', type=click.Choice(['csv', 'xml']), default='csv',
              help='Output format for metadata (csv or xml)')
@click.option('--output-file', type=click.Path(), default=None,
              help='Path to save the output file. Defaults to metadata.[format] in the input directory')
@click.option('--model', default=None, help='Model to use for processing (default: gemini-2.0-flash-exp)')
@click.option('--batch-size', type=int, default=None, help='Number of images to process in parallel')
@click.option('--no-compress', is_flag=True, help='Disable image compression (not recommended for large files)')
@click.option('--skip-existing', is_flag=True, help='Skip files that already have metadata in the output format')
async def generate_metadata(directory: str, api_key: str, output_format: str, output_file: str,
                          model: str, batch_size: int, no_compress: bool, skip_existing: bool):
    """Generate metadata files (CSV/XML) for images without modifying the original files."""
    try:
        # Initialize processor with configuration values as defaults
        processor = ImageProcessor(
            api_key=api_key,
            model=model,
            batch_size=batch_size,
            rename_files=False,  # Never rename files in metadata-only mode
            prefix=None
        )
        
        # Determine output path
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = Path(directory) / f"metadata.{output_format}"
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check for existing metadata if skip_existing is enabled
        existing_metadata = set()
        if skip_existing and output_path.exists():
            try:
                if output_format == 'csv':
                    df = pd.read_csv(output_path)
                    existing_metadata = set(df['path'].tolist())
                else:  # xml
                    tree = ET.parse(output_path)
                    root = tree.getroot()
                    existing_metadata = set(img.find('path').text for img in root.findall('image'))
                logging.info(f"Found {len(existing_metadata)} existing entries in {output_path}")
            except Exception as e:
                logging.warning(f"Error reading existing metadata file: {str(e)}")
        
        # Process images
        click.echo(f"Processing images in {directory}...")
        compress = not no_compress if no_compress is not None else config.compression_enabled
        
        # Filter out existing entries if skip_existing is enabled
        image_files = processor._get_image_files(directory)
        if skip_existing:
            original_count = len(image_files)
            image_files = [f for f in image_files if str(f) not in existing_metadata]
            skipped_count = original_count - len(image_files)
            if skipped_count > 0:
                click.echo(f"Skipping {skipped_count} files that already have metadata")
        
        results = await processor.process_images(directory, compress=compress, files=image_files)
        
        # Save results
        if results:
            if output_format == 'csv':
                if skip_existing and output_path.exists():
                    # Append to existing CSV
                    existing_df = pd.read_csv(output_path)
                    new_df = pd.DataFrame(results)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df.to_csv(output_path, index=False)
                else:
                    processor.save_to_csv(results, output_path)
            else:
                if skip_existing and output_path.exists():
                    # Merge with existing XML
                    tree = ET.parse(output_path)
                    root = tree.getroot()
                    processor.append_to_xml(results, root)
                    tree.write(output_path)
                else:
                    processor.save_to_xml(results, output_path)
            click.echo(f"Metadata saved to {output_path}")
        else:
            click.echo("No new images were processed")
        
        click.echo("Processing complete!")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    cli() 