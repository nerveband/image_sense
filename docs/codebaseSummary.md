# Codebase Summary v0.1.0

## Key Components and Their Interactions

### Core Components
-   **CLI (src/cli)**: Command-line interface for user interaction and process orchestration
-   **Models (src/models)**: AI model integrations (Gemini and Anthropic) for image analysis
-   **Output Handlers (src/output)**: Formatters for CSV and XML output with schema support
-   **Core (src/core)**: Core processing logic and utilities
-   **Tests (tests/)**: Comprehensive test suite using pytest

## Data Flow

1. User inputs command via CLI with configuration options
2. CLI validates input and determines processing mode (single/bulk)
3. Core processor loads and validates images
4. Selected AI model(s) generate metadata for each image
5. Output handlers format metadata according to schema
6. Results are written to specified location with error handling

## External Dependencies

### AI Models
-   **Gemini**: Primary model for image analysis
    - Rate limited API calls
    - Async processing support
-   **Anthropic**: Secondary model for enhanced analysis
    - Rate limited API calls
    - Async processing support

### Core Libraries
-   **pytest**: Testing framework
-   **click**: CLI framework
-   **pandas**: Data handling
-   **lxml**: XML processing
-   **Pillow**: Image processing
-   **aiohttp**: Async HTTP
-   **asyncio**: Async operations

## Recent Significant Changes

### v0.1.0 (2024-12-25)
-   Initial project structure
-   Basic CLI implementation
-   Core processing logic
-   AI model integration
-   Documentation setup
-   Version control implementation

## User Feedback Integration

-   Project in alpha stage
-   Feedback system to be implemented
-   GitHub issues tracking planned