# AI Image Processor

A powerful command-line tool for processing images with AI-powered metadata generation.

## Features

- Process multiple images simultaneously
- Generate accurate image descriptions and keywords using Google's Gemini AI
- Support for multiple image formats (JPEG, PNG, WebP)
- Export metadata in various formats (CSV, XML)
- Command-line interface for automation
- Cross-platform support (Windows, macOS, Linux)

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-processor.git
cd image-processor
```

2. Set up Python environment:

On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Gemini API key
# On macOS/Linux:
nano .env
# On Windows:
notepad .env
```

## Usage

### Command Line Interface

Process a single image:
```bash
python -m src.cli.main process --input path/to/image.jpg
```

Process a directory of images:
```bash
python -m src.cli.main process-batch --input path/to/images/
```

Export options:
```bash
python -m src.cli.main process --input image.jpg --export csv,xml --output-dir ./output
```

## Development

### Project Structure
```
image_processor/
├── src/
│   ├── core/           # Core processing modules
│   └── cli/            # Command-line interface
├── tests/              # Test files
├── resources/          # Application resources
│   └── exiftool/       # Bundled ExifTool
└── docs/              # Documentation
```

### Running Tests
```bash
# Activate virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run tests with coverage report
pytest
```

## Configuration

The application can be configured through:
- Environment variables (see `.env.example`)
- Command-line arguments

### Environment Variables

- `GOOGLE_API_KEY`: Your Gemini API key (required)
- `LOG_LEVEL`: Logging level (default: INFO)
- `BATCH_SIZE`: Number of images to process in a batch (default: 100)
- `ENABLE_COMPRESSION`: Enable image compression (default: true)
- `MAX_RETRIES`: Maximum number of retries for failed API calls (default: 3)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Troubleshooting

1. If image processing fails:
   - Verify your API key is correctly set
   - Check the supported image formats
   - Ensure you have internet connectivity
   - Look for error messages in the console

2. For development issues:
   - Check the logs
   - Run tests to verify functionality

## License

[MIT License](LICENSE)

## Acknowledgments

- Google Gemini AI for image analysis
- ExifTool for metadata management 