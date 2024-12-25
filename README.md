# Image Sense

<p align="center">
  <img src="icon/Image Sense Icon.png" alt="Image Sense Logo" width="200"/>
</p>

A powerful command-line tool for generating intelligent image metadata using Google's Gemini Vision AI.

## Features

- Generate detailed image descriptions and metadata
- Support for single image and bulk processing
- Multiple output formats (CSV, XML)
- Recursive directory processing
- Smart image compression (enabled by default)
- Efficient batch processing with configurable batch sizes
- Progress tracking for bulk operations
- Structured output with rich metadata

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-sense.git
cd image-sense
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment:
   - Create a `.env` file in the project root
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## Command Line Usage

The tool provides two main commands:

### 1. Process Single Image

```bash
image_sense process IMAGE_PATH [OPTIONS]
```

Options:
- `-f, --output-format [csv|xml]`: Output format for metadata (default: csv)
- `--no-compress`: Disable smart image compression (not recommended for large files)

Example:
```bash
image_sense process path/to/image.jpg --output-format xml
```

### 2. Bulk Process Directory

```bash
image_sense bulk-process DIRECTORY [OPTIONS]
```

Options:
- `-f, --output-format [csv|xml]`: Output format for metadata (default: csv)
- `-r, --recursive`: Process directories recursively
- `--no-compress`: Disable smart image compression (not recommended)
- `-b, --batch-size N`: Number of images to process in one batch (default: 1)
- `-m, --model MODEL`: Specify Gemini model to use

Example:
```bash
image_sense bulk-process path/to/directory --output-format csv --recursive --batch-size 4
```

## Output Formats

### CSV Format
The CSV output includes:
- File path
- Image description
- Technical details
- Visual elements
- Colors and composition
- Mood/atmosphere
- Potential use cases
- Keywords

### XML Format
The XML output provides the same information in a structured XML format with tags for each metadata category, including:
- Basic image information
- Technical details
- Visual elements
- Composition analysis
- Keywords and tags
- Mood and atmosphere
- Use cases

## Smart Compression

By default, the tool uses smart image compression to optimize processing:
- Automatically compresses large images (>10MB)
- Maintains visual quality while reducing processing time
- Can be disabled with `--no-compress` flag
- Typically provides 2-5x speedup for large images

## Requirements

- Python 3.8 or higher
- Google API key with Gemini Vision API access
- Supported image formats: JPG, JPEG, PNG, WebP
- ExifTool (bundled with the application)

## Development

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for development setup and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details 