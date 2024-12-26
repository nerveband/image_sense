# Image Sense v0.1.1

![Image Sense](icon/Image%20Sense%20Banner.jpg)

A powerful image analysis and metadata management tool powered by Google's Gemini Vision API.

## Status: Alpha Release
**CURRENTLY IN ALPHA. USE AT YOUR OWN RISK.**
- Version: 0.1.1
- Release Date: 2024-03-20
- Status: Development

## Features

- 🖼️ Advanced image analysis using Google's Gemini Vision API
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

Image Sense uses a flexible configuration system with a clear priority order:

1. **Command Line Arguments** (Highest Priority)
   - Arguments passed directly to commands override all other settings
   - Example: `--model gemini-2.0-flash-exp` overrides both .env and defaults

2. **Environment Variables** (.env file)
   - Settings in your .env file override default values
   - Won't override command line arguments

3. **Default Values** (Lowest Priority)
   - Built-in defaults used when no other value is specified
   - Example: `DEFAULT_MODEL=gemini-2.0-flash-exp`

### Command Line Arguments

Most environment variables can be overridden via command line arguments:

```bash
# Override model selection
image_sense process image.jpg --model gemini-2.0-flash-exp

# Override API key
image_sense process image.jpg --api-key YOUR_KEY

# Override output settings
image_sense process image.jpg --output-format xml --output-dir custom_output

# Override CSV behavior
image_sense process image.jpg --auto-append  # Append to existing CSV files

# Override processing options
image_sense process image.jpg --verbose --no-backup --duplicate
```

### Environment Variables

Configuration uses a priority system:
1. Command line arguments (highest priority)
2. Environment variables in `.env` file
3. Default values in code (lowest priority)

#### Image Processing
```env
# Enable smart compression (default: true)
COMPRESSION_ENABLED=true
# JPEG quality 1-100 (default: 85)
COMPRESSION_QUALITY=85
# Maximum dimension in pixels (default: 1024)
MAX_DIMENSION=1024
```

#### Batch Processing
```env
# Number of images to process in parallel (default: 50)
# Recommended range: 50-500 for optimal performance
DEFAULT_BATCH_SIZE=50
# Maximum batch size, Gemini API limit (default: 3000)
MAX_BATCH_SIZE=3000
# Minimum batch size for parallel processing (default: 10)
MIN_BATCH_SIZE=10
# Enable automatic batch optimization (default: true)
AUTO_OPTIMIZE_BATCH=true
# Maximum memory per batch in MB, 0 for unlimited (default: 1024)
MAX_BATCH_MEMORY=1024
```

#### Output Settings
```env
# Output format: 'csv' or 'xml' (default: csv)
DEFAULT_OUTPUT_FORMAT=csv
# Output directory (default: output)
OUTPUT_DIRECTORY=output
# Save CSV files (default: true)
SAVE_CSV_OUTPUT=true
# Save XML files (default: true)
SAVE_XML_OUTPUT=true
# Append to existing CSV files (default: false)
AUTO_APPEND_CSV=false
```

#### Model Settings
```env
# Default AI model (default: gemini-2.0-flash-exp)
DEFAULT_MODEL=gemini-2.0-flash-exp
# Available models:
# - gemini-2.0-flash-exp: Latest experimental model (fastest)
# - gemini-1.5-flash: Production model (balanced)
# - gemini-1.5-pro: More detailed analysis (slower)
```

#### File Handling
```env
# Enable automatic file renaming (default: false)
RENAME_FILES=false
# Prefix for renamed files (default: empty)
FILE_PREFIX=
# Create duplicates before modifying (default: false)
DUPLICATE_FILES=false
# Suffix for duplicate files (default: _modified)
DUPLICATE_SUFFIX=_modified
```

#### Metadata Settings
```env
# Create backup copies before modifying (default: true)
BACKUP_METADATA=true
# Write analysis to EXIF data (default: true)
WRITE_EXIF=true
```

#### Progress and Logging
```env
# Show progress bars and statistics (default: true)
SHOW_PROGRESS=true
# Show real-time model responses (default: true)
VERBOSE_OUTPUT=true
# Logging level (default: INFO)
LOG_LEVEL=INFO
```

#### API Settings
```env
# Google API key for Gemini Vision API (REQUIRED)
GOOGLE_API_KEY=your-google-api-key-here
```

#### System Settings
```env
# Path to ExifTool (default: resources/exiftool/exiftool)
EXIFTOOL_PATH=resources/exiftool/exiftool
# Maximum API call retries (default: 3)
MAX_RETRIES=3
# API call timeout in seconds (default: 30)
API_TIMEOUT=30
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

### Enhanced Batch Processing (March 2024)

1. **Optimized Batch Processing**
   - Now supports processing up to 3000 images per batch
   - Default batch size increased to 50 images
   - Automatic batch size optimization based on system resources
   - Intermediate results saving to prevent data loss
   - Memory usage management for large batches

2. **Configuration Options**
   ```env
   # Batch size configuration
   DEFAULT_BATCH_SIZE=50      # Default batch size (recommended: 50-500)
   MAX_BATCH_SIZE=3000       # Maximum batch size (Gemini limit)
   MIN_BATCH_SIZE=10         # Minimum batch size
   AUTO_OPTIMIZE_BATCH=true  # Enable automatic optimization
   MAX_BATCH_MEMORY=1024     # Maximum memory per batch (MB)
   ```

3. **Performance Improvements**
   - Parallel image optimization within batches
   - Efficient memory management for large batches
   - Progress tracking per batch and overall
   - Automatic cleanup of temporary files
   - Error handling and recovery for batch operations

4. **Usage Examples**
   ```bash
   # Process with default batch size (50)
   image_sense bulk-process path/to/directory

   # Process with custom batch size
   image_sense bulk-process path/to/directory --batch-size 100

   # Process with automatic batch optimization
   image_sense bulk-process path/to/directory --auto-optimize

   # Process with memory limit
   image_sense bulk-process path/to/directory --max-memory 2048
   ```

### Usage Examples

Process a single image with default settings (verbose):
```bash
image_sense process path/to/image.jpg
```

Process a single image and append results to existing CSV:
```bash
image_sense process path/to/image.jpg --auto-append
```

Process multiple images and append to existing CSV:
```bash
image_sense bulk-process path/to/directory --auto-append
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
  - Original path and filename (now included in Gemini output)
  - New filename (if renamed)
  - Modified path (if duplicated)
  - Suggested filename
- Image metadata
- Analysis results
- Technical information
- Visual characteristics

The XML output can be controlled via:
- Environment variable: `SAVE_XML_OUTPUT=true/false` (defaults to true)
- Output directory: Matches the folder name being processed with "_metadata.xml" suffix
- Each image analysis includes the original filename for proper tracking
- XML files are saved alongside CSV files when enabled

Example XML structure:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<image_analysis>
    <original_filename>example.jpg</original_filename>
    <description>Image description</description>
    <keywords>
        <keyword>term1</keyword>
        <keyword>term2</keyword>
    </keywords>
    <technical_details>
        <format>JPEG</format>
        <dimensions>1920x1080</dimensions>
        <color_space>sRGB</color_space>
    </technical_details>
    <visual_elements>
        <element>element1</element>
        <element>element2</element>
    </visual_elements>
    <composition>
        <technique>technique1</technique>
        <technique>technique2</technique>
    </composition>
    <mood>Mood description</mood>
    <use_cases>
        <use_case>use case 1</use_case>
        <use_case>use case 2</use_case>
    </use_cases>
    <suggested_filename>descriptive_name.jpg</suggested_filename>
</image_analysis>
```

Note: When processing a folder, the XML output file will be named after the folder (e.g., "folder_name_metadata.xml") and saved in the configured output directory.

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