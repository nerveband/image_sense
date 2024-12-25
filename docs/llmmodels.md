# Developer Documentation for Anthropic Claude and Google Gemini Models

**Default Models:**

*   **Gemini:** [gemini-2.0-flash-exp](https://ai.google.dev/gemini-api/docs/models/gemini)
*   **Claude:** [claude-3-5-haiku-20241022](https://docs.anthropic.com/en/docs/about-claude/models)

This document provides a comprehensive overview of the models offered by Anthropic (Claude) and Google (Gemini), extracted from the provided documentation. It includes model names, default model selections, and estimated pricing for image processing.

## Anthropic Claude Models

Claude is a family of state-of-the-art large language models developed by Anthropic. For more details, see the [Anthropic Claude Models Documentation](https://docs.anthropic.com/en/docs/about-claude/models). You can also find the Python SDK here: [anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python).

### Available Models

**Claude 3.5 Family:**

*   **Claude 3.5 Sonnet:** `claude-3-5-sonnet-20241022` (Anthropic API), `anthropic.claude-3-5-sonnet-20241022-v2:0` (AWS Bedrock), `claude-3-5-sonnet-v2@20241022` (GCP Vertex AI) -  Our most intelligent model.
*   **Claude 3.5 Haiku:** `claude-3-5-haiku-20241022` (Anthropic API), `anthropic.claude-3-5-haiku-20241022-v1:0` (AWS Bedrock), `claude-3-5-haiku@20241022` (GCP Vertex AI) - Our fastest model.

**Claude 3 Family:**

*   **Claude 3 Opus:** `claude-3-opus-20240229` (Anthropic API), `anthropic.claude-3-opus-20240229-v1:0` (AWS Bedrock), `claude-3-opus@20240229` (GCP Vertex AI) - Powerful model for highly complex tasks.
*   **Claude 3 Sonnet:** `claude-3-sonnet-20240229` (Anthropic API), `anthropic.claude-3-sonnet-20240229-v1:0` (AWS Bedrock), `claude-3-sonnet@20240229` (GCP Vertex AI) - Balance of intelligence and speed.
*   **Claude 3 Haiku:** `claude-3-haiku-20240307` (Anthropic API), `anthropic.claude-3-haiku-20240307-v1:0` (AWS Bedrock), `claude-3-haiku@20240307` (GCP Vertex AI) - Fastest and most compact model for near-instant responsiveness.

**Legacy Models:**

*   **Claude 2.1:** `claude-2.1`
*   **Claude 2:** `claude-2.0`
*   **Claude Instant 1.2:** `claude-instant-1.2`

### Default Model

The default Claude model is: `claude-3-5-haiku-20241022`.

## Google Gemini Models

The Gemini API offers different models optimized for specific use cases. See the [Gemini API Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for more information. You can find the Python SDK here: [google-gemini generative-ai-python](https://github.com/google-gemini/generative-ai-python).

### Available Models

**Current Models:**

*   **Gemini 2.0 Flash (Experimental):** `gemini-2.0-flash-exp`
*   **Gemini 1.5 Flash:** `gemini-1.5-flash`
*   **Gemini 1.5 Flash-8B:** `gemini-1.5-flash-8b`
*   **Gemini 1.5 Pro:** `gemini-1.5-pro`

**Deprecated Model:**

*   **Gemini 1.0 Pro (Deprecated on 2/15/2025):** `gemini-1.0-pro`

**Embedding Models:**

*   **Text Embedding:** `text-embedding-004`
*   **Embedding:** `embedding-001`

**Attributed Question Answering Model:**

*   **AQA:** `aqa`

### Default Model

The default Gemini model is: `gemini-2.0-flash-exp`.

## Estimated Pricing for 100 Images

To estimate the pricing for processing 100 images, we need to consider the input costs for models that support vision capabilities.

### Gemini Pricing

The provided documentation for Gemini does **not** explicitly state the pricing for image inputs. While `gemini-2.0-flash-exp`, `gemini-1.5-flash`, `gemini-1.5-flash-8b`, and `gemini-1.5-pro` support image inputs, the pricing is based on tokens for text. **Therefore, we cannot accurately estimate the cost for 100 images for Gemini models based on the provided information.**  You would need to consult the official Gemini API pricing page for details on image processing costs. See the [Gemini API Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for potential pricing updates.

### Claude Pricing

Only the following Claude models support vision (image input):

*   Claude 3.5 Sonnet
*   Claude 3 Opus
*   Claude 3 Sonnet

The default Claude model is `claude-3-5-haiku-20241022`, which **does not support vision**. Therefore, it cannot be used for image processing.

To estimate the cost for 100 images, we would need to use a vision-capable Claude model. However, the pricing provided in the document is in terms of text input and output tokens (per MTok). **There is no explicit pricing for image input in the Claude documentation provided.** Please refer to the [Anthropic Claude Models Documentation](https://docs.anthropic.com/en/docs/about-claude/models) for any potential updates on image pricing.

**Conclusion on Image Pricing:**

Based on the provided documentation, we can't directly calculate the cost of processing 100 images for either Gemini or Claude using their default models.

*   **Gemini:** The documentation doesn't provide specific pricing for image inputs.
*   **Claude:** The default model (`claude-3-5-haiku-20241022`) does not support vision. Even for vision-capable Claude models, the pricing is based on text tokens, not image inputs.

To get accurate pricing for image processing, please refer to the official pricing pages for the respective APIs:

*   **Anthropic Claude Pricing:**  Likely available on the Anthropic website or developer portal.
*   **Google Gemini Pricing:**  Available on the Google Cloud or Google AI for Developers website.