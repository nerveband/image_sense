# Developer Documentation for Google Gemini Models

**Default Model:**

* **Gemini:** [gemini-2.0-flash-exp](https://ai.google.dev/gemini-api/docs/models/gemini)

This document provides a comprehensive overview of the models offered by Google (Gemini), extracted from the provided documentation. It includes model names, default model selections, and estimated pricing for image processing.

## Google Gemini Models

The Gemini API offers different models optimized for specific use cases. See the [Gemini API Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for more information. You can find the Python SDK here: [google-gemini generative-ai-python](https://github.com/google-gemini/generative-ai-python).

### Available Models

**Current Models:**

* **Gemini 2.0 Flash (Experimental):** `gemini-2.0-flash-exp`
* **Gemini 1.5 Flash:** `gemini-1.5-flash`
* **Gemini 1.5 Flash-8B:** `gemini-1.5-flash-8b`
* **Gemini 1.5 Pro:** `gemini-1.5-pro`

**Deprecated Model:**

* **Gemini 1.0 Pro (Deprecated on 2/15/2025):** `gemini-1.0-pro`

**Embedding Models:**

* **Text Embedding:** `text-embedding-004`
* **Embedding:** `embedding-001`

**Attributed Question Answering Model:**

* **AQA:** `aqa`

### Default Model

The default Gemini model is: `gemini-2.0-flash-exp`.

## Estimated Pricing for 100 Images

Based on the Vertex AI pricing documentation, we can calculate the cost for processing 100 images using different Gemini models:

### Gemini 1.5 Flash
- Standard context (≤ 128K input tokens): $0.00002 per image
- Long context (> 128K input tokens): $0.00004 per image
- **Cost for 100 images:** 
  - Standard context: $0.002 (100 × $0.00002)
  - Long context: $0.004 (100 × $0.00004)

### Gemini 1.5 Pro
- Standard context (≤ 128K input tokens): $0.00032875 per image
- Long context (> 128K input tokens): $0.0006575 per image
- **Cost for 100 images:**
  - Standard context: $0.032875 (100 × $0.00032875)
  - Long context: $0.06575 (100 × $0.0006575)

### Gemini 1.0 Pro
- Fixed rate: $0.0025 per image
- **Cost for 100 images:** $0.25 (100 × $0.0025)

**Notes:**
- Prices are listed in US Dollars (USD)
- Batch processing is available at a 50% discount
- PDFs are billed as image input, with one PDF page equivalent to one image