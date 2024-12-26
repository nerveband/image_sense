"""Command line interface for image processing."""
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from core.config import config
from core.llm_handler import get_provider

def main():
    """Main CLI entrypoint."""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Process images using Gemini Vision.')
    parser.add_argument('input', help='Input image or directory path')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Enable verbose output (overrides VERBOSE_OUTPUT env var)')
    parser.add_argument('-b', '--batch-size', type=int,
                       help=f'Batch size for processing (default: {config.default_batch_size})')
    parser.add_argument('-o', '--output-dir',
                       help=f'Output directory (default: {config.output_directory})')
    parser.add_argument('-f', '--format', choices=['csv', 'json'],
                       help=f'Output format (default: {config.default_output_format})')
    
    args = parser.parse_args()
    
    # Override config with command line arguments if provided
    if args.verbose:
        os.environ['VERBOSE_OUTPUT'] = 'true'
        config.verbose_output = True
    if args.batch_size:
        config.default_batch_size = args.batch_size
    if args.output_dir:
        config.output_directory = args.output_dir
    if args.format:
        config.default_output_format = args.format
    
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")
    
    # Initialize provider with verbose setting from config
    provider = get_provider(api_key, config.verbose_output)
    
    # Process input path
    input_path = Path(args.input)
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")
    
    # TODO: Add processing logic here
    
if __name__ == '__main__':
    main()
