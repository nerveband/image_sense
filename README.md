# Image Sense

![Image Sense](icon/Image%20Sense%20Banner.jpg)

An AI-powered image analysis and metadata management tool that uses state-of-the-art machine learning models to analyze images and generate rich, structured metadata.

## Features

- 🖼️ Advanced image analysis using Google's Gemini Vision API
- 📝 Rich, structured metadata generation
- 🔄 Batch processing with smart compression
- 💾 Multiple output formats (CSV, XML)
- 🏷️ Automatic EXIF metadata writing
- 📊 Progress tracking and statistics
- ⚙️ Highly configurable via environment variables

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
```

#### Progress and Logging
```env
# Show progress bars and statistics
SHOW_PROGRESS=true
# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO
```

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
The `generate-metadata` command analyzes images and creates structured metadata files without modifying the originals:
```bash
image_sense generate-metadata path/to/directory --api-key YOUR_API_KEY [OPTIONS]
```

Key features:
- Non-destructive: Original images remain unchanged
- Flexible output: Choose between CSV and XML formats
- Smart compression: Optimized for faster processing
- Batch processing: Handle multiple images efficiently
- Incremental updates: Skip already processed files

Options:
- `--output-format, -f`: Choose output format (csv/xml)
- `--output-file`: Specify custom output file path
- `--model`: Select AI model to use
- `--batch-size`: Set custom batch size
- `--no-compress`: Disable image compression
- `--skip-existing`: Skip files that already have metadata

Example with incremental processing:
```bash
# First run - process all images
image_sense generate-metadata photos/ --api-key YOUR_API_KEY

# Later runs - only process new images
image_sense generate-metadata photos/ --api-key YOUR_API_KEY --skip-existing
```

For detailed command documentation, see [Commands Documentation](docs/commands.md).

## Output Formats

### CSV Format
The CSV output includes columns for:
- File path
- Description
- Keywords
- Technical details
- Visual elements
- Composition
- Mood
- Use cases

### XML Format
The XML output provides a structured representation of:
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