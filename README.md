# Image Sense v0.1.0

![Image Sense](icon/Image%20Sense%20Banner.jpg)

A powerful image analysis and metadata management tool powered by Google's Gemini Vision API.

## Status: Alpha Release
**CURRENTLY IN ALPHA. USE AT YOUR OWN RISK.**
- Version: 0.1.0
- Release Date: 2024-12-25
- Status: Development

## Features

- 🖼️ Advanced image analysis using Google's Gemini Vision API and Anthropic Claude
- 📝 Rich, structured metadata generation with AI-powered descriptions
- 🔄 Batch processing with smart compression and parallel processing
- 💾 Multiple output formats (CSV, XML) with customizable schemas
- 🏷️ Automatic EXIF metadata writing and management
- 📊 AI-powered filename suggestions and organization
- 📋 Complete file operation tracking with detailed logs
- 🔒 Non-destructive processing with backup options
- 📊 Progress tracking and detailed statistics
- ⚙️ Highly configurable via environment variables and CLI

## Installation

1. Ensure you have Python 3.8 or higher installed

2. Clone the repository:
```bash
git clone https://github.com/nerveband/image_sense.git
cd image_sense
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

5. Copy the example environment file and configure your settings:
```bash
cp .env.example .env
```

6. Edit `.env` with your API keys and preferences

## Configuration

Image Sense can be configured using environment variables. Create a `.env` file in the project root with the following options:

### Default Values and Configuration

Below are the default values used by the application. You can override any of these in your `.env` file:

#### Image Processing
```env
# Enable smart compression (recommended for large files)
COMPRESSION_ENABLED=true
# JPEG quality (1-100, higher = better quality but larger size)
COMPRESSION_QUALITY=85
# Maximum dimension in pixels for processing
MAX_DIMENSION=1024
```

#### Batch Processing
```env
# Number of images to process in parallel
DEFAULT_BATCH_SIZE=8
# Maximum allowed batch size (model-dependent)
MAX_BATCH_SIZE=16
```

#### Output Settings
```env
# Default output format (csv or xml)
DEFAULT_OUTPUT_FORMAT=csv
# Directory for output files
OUTPUT_DIRECTORY=output
```

#### Model Settings
```env
# Default AI model
DEFAULT_MODEL=gemini-2.0-flash-exp
# Available models:
# - gemini-2.0-flash-exp: Latest experimental model (fastest)
# - gemini-1.5-flash: Production model (balanced)
# - gemini-1.5-pro: More detailed analysis (slower)
```

#### Metadata Settings
```env
# Create backups before modifying metadata
BACKUP_METADATA=true
# Write analysis results to image EXIF data
WRITE_EXIF=true
# Create duplicate files before modifying
DUPLICATE_FILES=false
# Suffix for duplicate files
DUPLICATE_SUFFIX=_modified
```

#### Progress and Logging
```env
# Show progress bars and statistics
SHOW_PROGRESS=true
# Show real-time Gemini model responses
VERBOSE_OUTPUT=false
# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO
```

## Recent Updates

### Enhanced Image Analysis and Feedback (December 25, 2023)

1. **Improved Verbose Output (Now Default)**
   - Verbose output is now enabled by default for better visibility
   - Added `--verboseoff` flag to disable verbose output if needed
   - Added detailed progress indicators for:
     - Image optimization/compression
     - Gemini API interactions
     - XML parsing and validation
     - CSV output generation

2. **Enhanced Image Processing**
   - Added automatic image optimization for Gemini API
   - Shows compression statistics (original size, compressed size, reduction percentage)
   - Better error handling for image processing failures

3. **Improved Data Handling**
   - Better XML parsing with proper Unicode support
   - Enhanced CSV output with all fields properly populated
   - Added suggested filename to output
   - Fixed various edge cases in data extraction

4. **Configuration Updates**
   - Environment variables are now properly respected (VERBOSE_OUTPUT, GOOGLE_API_KEY, etc.)
   - Verbose output can be controlled via:
     - Environment variable: `VERBOSE_OUTPUT=false`
     - CLI flag: `--verboseoff`
     - Default is verbose on

### Usage Examples

Process a single image with default settings (verbose):
```bash
image_sense process path/to/image.jpg
```

Process a single image with verbose output disabled:
```bash
image_sense process path/to/image.jpg --verboseoff
```

Bulk process a directory of images:
```bash
image_sense bulk-process path/to/directory
```

Bulk process with specific options:
```bash
image_sense bulk-process path/to/directory --recursive --verboseoff --model 2-flash
```

### Environment Variables

- `VERBOSE_OUTPUT`: Control verbose output (default: "true")
- `GOOGLE_API_KEY`: Your Google API key for Gemini
- `GEMINI_MODEL`: Default model to use (default: "gemini-2.0-flash-exp")

## API Keys

You'll need a Google API key with Gemini Vision API access enabled:
1. Get it from: https://aistudio.google.com/app/apikey
2. Add it to your `.env` file as `GOOGLE_API_KEY=your-key-here`
3. Or pass it directly using the `--api-key` parameter

## Usage

### Quick Start

1. Generate metadata for a directory of images:
```bash
image_sense generate-metadata path/to/photos --api-key YOUR_API_KEY
```
This will analyze all images and create a metadata.csv file with detailed descriptions, keywords, and technical details.

2. Process a single image:
```bash
image_sense process path/to/image.jpg
```

3. Process multiple images with advanced options:
```bash
image_sense bulk-process path/to/directory --api-key YOUR_API_KEY --output-format xml
```

### Command Options

#### Generate Metadata (Recommended)
The `generate-metadata` command analyzes images and creates structured metadata files:
```bash
image_sense generate-metadata path/to/directory --api-key YOUR_API_KEY [OPTIONS]
```

Key features:
- Non-destructive: Original images remain unchanged
- Flexible output: Choose between CSV and XML formats
- Smart compression: Optimized for faster processing
- Batch processing: Handle multiple images efficiently
- Incremental updates: Skip already processed files
- AI-powered filename suggestions
- Complete file operation tracking

Options:
- `--output-format, -f`: Choose output format (csv/xml)
- `--output-file`: Specify custom output file path
- `--model`: Select AI model to use
- `--batch-size`: Set custom batch size
- `--no-compress`: Disable image compression
- `--skip-existing`: Skip files that already have metadata
- `--duplicate`: Create duplicates before modifying files
- `--no-backup`: Disable ExifTool backup creation

Example with duplicate files:
```bash
# Process images and create duplicates before modification
image_sense generate-metadata photos/ --api-key YOUR_API_KEY --duplicate

# Process without creating duplicates (modify in place)
image_sense generate-metadata photos/ --api-key YOUR_API_KEY
```

For detailed command documentation, see [Commands Documentation](docs/commands.md).

## Output Formats

### CSV Format
The CSV output includes columns for:
- Original file path
- Original filename
- New filename (if renamed)
- Modified file path (if duplicated)
- Suggested filename
- Description
- Keywords
- Technical details
- Visual elements
- Composition
- Mood
- Use cases

### XML Format
The XML output provides a structured representation of:
- File information
  - Original path and filename
  - New filename (if renamed)
  - Modified path (if duplicated)
  - Suggested filename
- Image metadata
- Analysis results
- Technical information
- Visual characteristics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Gemini Vision API for image analysis
- ExifTool for metadata management
- Rich for beautiful terminal output 