# Image Sense CLI Commands

This document provides detailed information about the available CLI commands in Image Sense.

## Commands Overview

### `process`
Process a single image and generate metadata.

```bash
image_sense process path/to/image.jpg [OPTIONS]
```

Options:
- `--output-format, -f`: Output format (csv/xml) [default: csv]
- `--no-compress`: Disable image compression

### `bulk-process`
Process multiple images in a directory.

```bash
image_sense bulk-process path/to/directory [OPTIONS]
```

Options:
- `--api-key`: (Required) Google API key
- `--model`: Model to use for processing
- `--batch-size`: Number of images to process in parallel
- `--no-compress`: Disable image compression
- `--output-format`: Output format (csv/xml)
- `--output-dir`: Directory to save output files
- `--rename-files`: Enable file renaming
- `--prefix`: Prefix for renamed files

### `generate-metadata`
Generate metadata files for images without modifying the originals.

```bash
image_sense generate-metadata path/to/directory [OPTIONS]
```

Options:
- `--api-key`: (Required) Google API key
- `--output-format, -f`: Output format (csv/xml) [default: csv]
- `--output-file`: Custom output file path
- `--model`: Model to use for processing
- `--batch-size`: Number of images to process in parallel
- `--no-compress`: Disable image compression
- `--skip-existing`: Skip files that already have metadata in the output format

## Detailed Command Documentation

### Generate Metadata Command

The `generate-metadata` command is designed for users who want to analyze their images and get structured metadata without modifying the original files. This is particularly useful for:
- Cataloging image collections
- Creating image databases
- Analyzing image content without altering files
- Generating reports about image collections

#### Usage Examples

1. Basic usage (generates metadata.csv in the input directory):
```bash
image_sense generate-metadata path/to/photos --api-key YOUR_API_KEY
```

2. Generate XML output:
```bash
image_sense generate-metadata path/to/photos --api-key YOUR_API_KEY --output-format xml
```

3. Save to a specific file:
```bash
image_sense generate-metadata path/to/photos --api-key YOUR_API_KEY --output-file results.csv
```

4. Use a specific model and batch size:
```bash
image_sense generate-metadata path/to/photos --api-key YOUR_API_KEY \
    --model gemini-2.0-flash-exp --batch-size 16
```

5. Skip files that already have metadata:
```bash
image_sense generate-metadata path/to/photos --api-key YOUR_API_KEY --skip-existing
```
This is useful for:
- Resuming interrupted processing
- Adding new images to an existing catalog
- Updating metadata selectively

#### Incremental Processing

When using the `--skip-existing` flag:
- The command checks for existing metadata files in the specified output format
- Files that already have entries in the metadata file are skipped
- New results are appended to the existing metadata file
- Progress information shows how many files were skipped

This feature is particularly useful for:
- Large image collections that are updated frequently
- Resuming interrupted processing sessions
- Saving time and API costs by avoiding reprocessing
- Maintaining a single metadata file for your entire collection

#### Output Format Details

##### CSV Output
The CSV file includes the following columns:
- `path`: Path to the original image
- `description`: Detailed description of the image
- `keywords`: Comma-separated list of relevant keywords
- `technical_details`: Image format, dimensions, and color space
- `visual_elements`: Key visual elements present in the image
- `composition`: Notable composition techniques used
- `mood`: Overall mood or atmosphere
- `use_cases`: Potential applications for the image

##### XML Output
The XML file provides a structured representation with the following hierarchy:
```xml
<images>
  <image>
    <path>path/to/image.jpg</path>
    <description>...</description>
    <keywords>
      <keyword>...</keyword>
      ...
    </keywords>
    <technical_details>
      <format>...</format>
      <dimensions>...</dimensions>
      <color_space>...</color_space>
    </technical_details>
    <visual_elements>
      <element>...</element>
      ...
    </visual_elements>
    <composition>
      <technique>...</technique>
      ...
    </composition>
    <mood>...</mood>
    <use_cases>
      <use_case>...</use_case>
      ...
    </use_cases>
  </image>
  ...
</images>
```

#### Performance Considerations

- **Compression**: By default, images are compressed before processing to improve speed and reduce API costs. Use `--no-compress` to disable this feature if image quality is critical.
- **Batch Size**: The default batch size is optimized for most use cases. Adjust with `--batch-size` based on your system's capabilities and API limits.
- **Model Selection**: Different models offer different trade-offs:
  - `gemini-2.0-flash-exp`: Fastest, good for large batches
  - `gemini-1.5-flash`: Balanced speed and accuracy
  - `gemini-1.5-pro`: Most detailed analysis but slower

#### Real-time Progress and Streaming

The command provides detailed real-time progress information:

- Overall Progress Bar:
  - Shows total progress across all images
  - Displays estimated time remaining
  - Shows processing speed

- Batch Progress:
  - Shows current batch being processed
  - Displays streaming updates as results are generated
  - Indicates batch number and size

- Status Updates:
  - Number of files skipped (with `--skip-existing`)
  - Success/failure counts
  - Error messages for failed images
  - Final statistics summary

Example output:
```
Processing 100 images...  [######################]  100%  10.2 imgs/s  ETA: 0:00
Batch 3/5...            [############--------]   60%  2.1 s remaining
```

The streaming functionality provides several benefits:
- See results as they're generated
- Early feedback on processing status
- Better progress tracking
- Immediate error reporting
- More responsive user experience 