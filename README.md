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

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image_sense.git
cd image_sense
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy the example environment file and configure your settings:
```bash
cp .env.example .env
```

4. Edit `.env` with your API keys and preferences

## Configuration

Image Sense can be configured using environment variables. Create a `.env` file in the project root with the following options:

### Image Processing
- `COMPRESSION_ENABLED`: Enable image compression (default: true)
- `COMPRESSION_QUALITY`: JPEG quality for compressed images (default: 85)
- `MAX_DIMENSION`: Maximum image dimension in pixels (default: 1024)

### Batch Processing
- `DEFAULT_BATCH_SIZE`: Number of images to process in parallel (default: 8)
- `MAX_BATCH_SIZE`: Maximum allowed batch size (default: 16)

### Output Configuration
- `DEFAULT_OUTPUT_FORMAT`: Default output format - 'csv' or 'xml' (default: csv)
- `OUTPUT_DIRECTORY`: Directory for output files (default: output)

### Model Configuration
- `DEFAULT_MODEL`: Default AI model to use (default: gemini-2.0-flash-exp)
- `RENAME_FILES`: Automatically rename processed files (default: false)
- `FILE_PREFIX`: Prefix for renamed files (default: empty)

### Metadata Configuration
- `BACKUP_METADATA`: Create backups before modifying metadata (default: true)
- `WRITE_EXIF`: Write analysis results to image EXIF data (default: true)

### Progress and Logging
- `SHOW_PROGRESS`: Show progress bars and statistics (default: true)
- `LOG_LEVEL`: Logging level (default: INFO)

## Usage

### Process a Single Image
```bash
image_sense process path/to/image.jpg
```

### Process a Directory of Images
```bash
image_sense bulk-process path/to/directory --output-format xml
```

### Options
- `--no-compress`: Disable image compression
- `--output-format`: Choose output format (csv/xml)
- `--batch-size`: Set custom batch size
- `--rename-files`: Enable file renaming
- `--prefix`: Set prefix for renamed files

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