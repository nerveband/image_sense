# Development Guide

This guide explains how to set up the development environment, build the application, and run tests.

## Prerequisites

- Python 3.11 or higher
- Git

## Development Environment Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd image-processor
   ```

2. Set up Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create environment file:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys (Gemini and Anthropic)
   ```

## Running the Application

### Command Line Usage

1. Process a single image:
   ```bash
   python -m src.cli.main process --input path/to/image.jpg
   ```

2. Process multiple images:
   ```bash
   python -m src.cli.main process-batch --input path/to/images/
   ```

3. Export options:
   ```bash
   python -m src.cli.main process --input image.jpg --export csv,xml --output-dir ./output
   ```

## Development Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and write tests.

3. Run tests:
   ```bash
   pytest
   ```

4. Format your code:
   ```bash
   black src/ tests/
   ```

5. Create a pull request.

## Project Structure

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

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src

# Run specific test file
pytest tests/test_bulk.py

# Run tests in parallel
pytest -n auto
```

### Test Data

- Test images are located in `tests/test_images/`
- Test metadata is stored in `tests/test_metadata/`

## Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Write descriptive variable names

## Error Handling

- Use appropriate exception types
- Log errors with context
- Provide helpful error messages to users
- Handle API rate limits and retries

## Documentation

- Keep README.md up to date
- Document all new features
- Update API documentation
- Include examples in docstrings

## Continuous Integration

- Tests run on every pull request
- Coverage reports are generated
- Code formatting is checked
- Dependencies are kept up to date

## Release Process

1. Update version number
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Create GitHub release
6. Update documentation

## Troubleshooting

### Common Issues

1. API Key Issues:
   - Check .env file configuration
   - Verify API key validity
   - Check API rate limits

2. Image Processing Issues:
   - Verify image format support
   - Check file permissions
   - Ensure ExifTool is accessible

3. Test Failures:
   - Check test data availability
   - Verify environment setup
   - Look for API mock issues

### Getting Help

- Check existing issues on GitHub
- Create a new issue with:
  - Clear description
  - Steps to reproduce
  - Expected vs actual behavior
  - Environment details 