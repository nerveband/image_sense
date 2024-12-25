## Key Components and Their Interactions

-   **CLI**: The command-line interface handles user input and orchestrates the processing.
-   **Models**: The AI models (Gemini and Anthropic) generate image metadata.
-   **Output Handlers**: These components format the metadata into CSV or XML.
-   **Tests**: Pytest is used to ensure the functionality of all components.

## Data Flow

1. User inputs command via CLI.
2. CLI parses the command and determines the processing mode (single or bulk).
3. Images are loaded from the specified directory.
4. The selected AI model generates metadata for each image.
5. Metadata is formatted into the chosen output format (CSV or XML).
6. Output is written to the specified file or directory.

## External Dependencies

-   **Gemini**: Used for generating image metadata.
-   **Anthropic**: Used for generating image metadata.
-   **pytest**: Used for testing.
-   **click**: Used for creating the command-line interface.
-   **pandas**: Used for handling CSV output.
-   **lxml**: Used for handling XML output.

## Recent Significant Changes

-   Initial project setup.
-   Basic structure for CLI, models, and output handlers.

## User Feedback Integration and Its Impact on Development

-   None yet. 