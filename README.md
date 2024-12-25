# Image Processor

A command-line tool for generating detailed metadata for images using Google's Gemini Vision AI.

## Features

- Generate detailed image descriptions and metadata
- Support for single image and bulk processing
- Multiple output formats (CSV, XML)
- Recursive directory processing
- Progress tracking for bulk operations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-processor.git
cd image-processor
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
image_processor process IMAGE_PATH [OPTIONS]
```

Options:
- `-f, --output-format [csv|xml]`: Output format for metadata (default: csv)

Example:
```bash
image_processor process path/to/image.jpg --output-format xml
```

### 2. Bulk Process Directory

```bash
image_processor bulk-process DIRECTORY [OPTIONS]
```

Options:
- `-f, --output-format [csv|xml]`: Output format for metadata (default: csv)
- `-r, --recursive`: Process directories recursively

Example:
```bash
image_processor bulk-process path/to/directory --output-format csv --recursive
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

### XML Format
The XML output provides the same information in a structured XML format with tags for each metadata category.

## Requirements

- Python 3.8 or higher
- Google API key with Gemini Vision API access
- Supported image formats: JPG, JPEG, PNG, WebP

## Development

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for development setup and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details 