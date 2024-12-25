# Technical Stack Documentation

## Technology Stack v2.0

### Core Technologies
- **Python 3.11+**: Primary language for CLI application

### Python Libraries
- **google-generativeai**: Gemini AI API integration
- **anthropic**: Claude AI API integration
- **Pillow**: Image processing and manipulation
- **aiohttp**: Async HTTP client for API calls
- **asyncio**: Async I/O operations
- **argparse**: CLI argument parsing
- **logging**: Built-in logging functionality
- **pandas**: Data manipulation and analysis
- **lxml**: XML processing

### Image Processing Tools
- **exiftool**: Bundled metadata manipulation tool
  - Will be packaged with the application
  - Platform-specific binaries included

### Data Serialization
- **csv**: Built-in CSV handling
- **xml.etree.ElementTree**: Built-in XML processing
- Custom schema definitions for structured output

### Development Tools
- **pytest**: Unit testing
- **Virtual Environment**: Python dependency isolation

### External Services
- **Google Gemini API**: AI-powered image analysis
  - Requires API key configuration
  - Rate limiting and retry mechanisms implemented
- **Anthropic Claude API**: Alternative AI model for image analysis
  - Requires API key configuration
  - Rate limiting and retry mechanisms implemented

### Cross-Platform Support
- Windows
- macOS
- Linux

### Version Control
- Git
- GitHub repository

### Development Environment
- VSCode recommended
- Black for Python formatting 

## AI Models

- Gemini
- Anthropic Claude

## File Formats

- CSV
- XML

## Testing

- pytest is used for writing and running tests.
- Test data is stored in `tests/test_images` and `tests/test_metadata`. 