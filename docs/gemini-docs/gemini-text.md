Title: Generate text using the Gemini API

URL Source: https://ai.google.dev/gemini-api/docs/text-generation?lang=python

Markdown Content:
The Gemini API can generate text output when provided text, images, video, and audio as input.

This guide shows you how to generate text using the [`generateContent`](https://ai.google.dev/api/rest/v1/models/generateContent) and [`streamGenerateContent`](https://ai.google.dev/api/rest/v1/models/streamGenerateContent) methods. To learn about working with Gemini's vision and audio capabilities, refer to the [Vision](https://ai.google.dev/gemini-api/docs/vision) and [Audio](https://ai.google.dev/gemini-api/docs/audio) guides.

Before you begin: Set up your project and API key
-------------------------------------------------

Before calling the Gemini API, you need to set up your project and configure your API key.

**Expand to view how to set up your project and API key**

### Get and secure your API key

You need an API key to call the Gemini API. If you don't already have one, create a key in Google AI Studio.

[Get an API key](https://aistudio.google.com/app/apikey)

It's strongly recommended that you do _not_ check an API key into your version control system.

You should store your API key in a secrets store such as Google Cloud [Secret Manager](https://cloud.google.com/secret-manager/docs).

This tutorial assumes that you're accessing your API key as an environment variable.

### Install the SDK package and configure your API key

The Python SDK for the Gemini API is contained in the [`google-generativeai`](https://pypi.org/project/google-generativeai/) package.

1.  Install the dependency using pip:
    
    ```
    pip install -U google-generativeai
    ```
    
2.  Import the package and configure the service with your API key:
    
    ```
    import os
    import google.generativeai as genai
    
    genai.configure(api_key=os.environ['API_KEY'])
    ```
    

Generate text from text-only input
----------------------------------

The simplest way to generate text using the Gemini API is to provide the model with a single text-only input, as shown in this example:

```
import google.generativeai as genai

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Write a story about a magic backpack.")
print(response.text)text_generation.py
```
In this case, the prompt ("Write a story about a magic backpack") doesn't include any output examples, system instructions, or formatting information. It's a [zero-shot](https://ai.google.dev/gemini-api/docs/models/generative-models#zero-shot-prompts) approach. For some use cases, a [one-shot](https://ai.google.dev/gemini-api/docs/models/generative-models#one-shot-prompts) or [few-shot](https://ai.google.dev/gemini-api/docs/models/generative-models#few-shot-prompts) prompt might produce output that's more aligned with user expectations. In some cases, you might also want to provide [system instructions](https://ai.google.dev/gemini-api/docs/system-instructions) to help the model understand the task or follow specific guidelines.

Generate text from text-and-image input
---------------------------------------

The Gemini API supports multimodal inputs that combine text with media files. The following example shows how to generate text from text-and-image input:

```
import google.generativeai as genai

import PIL.Image

model = genai.GenerativeModel("gemini-1.5-flash")
organ = PIL.Image.open(media / "organ.jpg")
response = model.generate_content(["Tell me about this instrument", organ])
print(response.text)text_generation.py
```
As with text-only prompting, multimodal prompting can involve various approaches and refinements. Depending on the output from this example, you might want to add steps to the prompt or be more specific in your instructions. To learn more, see [File prompting strategies](https://ai.google.dev/gemini-api/docs/file-prompting-strategies).

Generate a text stream
----------------------

By default, the model returns a response after completing the entire text generation process. You can achieve faster interactions by not waiting for the entire result, and instead use streaming to handle partial results.

The following example shows how to implement streaming using the [`streamGenerateContent`](https://ai.google.dev/api/rest/v1/models/streamGenerateContent) method to generate text from a text-only input prompt.

```
import google.generativeai as genai

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Write a story about a magic backpack.", stream=True)
for chunk in response:
    print(chunk.text)
    print("_" * 80)text_generation.py
```

Build an interactive chat
-------------------------

You can use the Gemini API to build interactive chat experiences for your users. Using the chat feature of the API lets you collect multiple rounds of questions and responses, allowing users to step incrementally toward answers or get help with multipart problems. This feature is ideal for applications that require ongoing communication, such as chatbots, interactive tutors, or customer support assistants.

The following code example shows a basic chat implementation:

```
import google.generativeai as genai

model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(
    history=[
        {"role": "user", "parts": "Hello"},
        {"role": "model", "parts": "Great to meet you. What would you like to know?"},
    ]
)
response = chat.send_message("I have 2 dogs in my house.")
print(response.text)
response = chat.send_message("How many paws are in my house?")
print(response.text)chat.py
```

Enable chat streaming
---------------------

You can also use streaming with chat, as shown in the following example:

```
import google.generativeai as genai

model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(
    history=[
        {"role": "user", "parts": "Hello"},
        {"role": "model", "parts": "Great to meet you. What would you like to know?"},
    ]
)
response = chat.send_message("I have 2 dogs in my house.", stream=True)
for chunk in response:
    print(chunk.text)
    print("_" * 80)
response = chat.send_message("How many paws are in my house?", stream=True)
for chunk in response:
    print(chunk.text)
    print("_" * 80)

print(chat.history)chat.py
```

Configure text generation
-------------------------

Every prompt you send to the model includes [parameters](https://ai.google.dev/gemini-api/docs/models/generative-models#model-parameters) that control how the model generates responses. You can use [`GenerationConfig`](https://ai.google.dev/api/rest/v1/GenerationConfig) to configure these parameters. If you don't configure the parameters, the model uses default options, which can vary by model.

The following example shows how to configure several of the available options.

```
import google.generativeai as genai

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(
    "Tell me a story about a magic backpack.",
    generation_config=genai.types.GenerationConfig(
        # Only one candidate for now.
        candidate_count=1,
        stop_sequences=["x"],
        max_output_tokens=20,
        temperature=1.0,
    ),
)

print(response.text)configure_model_parameters.py
```
`candidateCount` specifies the number of generated responses to return. Currently, this value can only be set to 1. If unset, this will default to 1.

`stopSequences` specifies the set of character sequences (up to 5) that will stop output generation. If specified, the API will stop at the first appearance of a `stop_sequence`. The stop sequence won't be included as part of the response.

`maxOutputTokens` sets the maximum number of tokens to include in a candidate.

`temperature` controls the randomness of the output. Use higher values for more creative responses, and lower values for more deterministic responses. Values can range from \[0.0, 2.0\].

You can also configure individual calls to `generateContent`:

```
response = model.generate_content(
    'Write a story about a magic backpack.',
    generation_config = genai.GenerationConfig(
        max_output_tokens=1000,
        temperature=0.1,
    )
)
```
Any values set on the individual call override values on the model constructor.

What's next
-----------

Now that you have explored the basics of the Gemini API, you might want to try:

*   [Vision understanding](https://ai.google.dev/gemini-api/docs/vision): Learn how to use Gemini's native vision understanding to process images and videos.
*   [System instructions](https://ai.google.dev/gemini-api/docs/system-instructions): System instructions let you steer the behavior of the model based on your specific needs and use cases.
*   [Audio understanding](https://ai.google.dev/gemini-api/docs/audio): Learn how to use Gemini's native audio understanding to process audio files.