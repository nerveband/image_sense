Title: Generate structured output with the Gemini API

URL Source: https://ai.google.dev/gemini-api/docs/structured-output?lang=python

Markdown Content:
Gemini generates unstructured text by default, but some applications require structured text. For these use cases, you can constrain Gemini to respond with JSON, a structured data format suitable for automated processing. You can also constrain the model to respond with one of the options specified in an enum.

Here are a few use cases that might require structured output from the model:

*   Build a database of companies by pulling company information out of newspaper articles.
*   Pull standardized information out of resumes.
*   Extract ingredients from recipes and display a link to a grocery website for each ingredient.

In your prompt, you can ask Gemini to produce JSON-formatted output, but note that the model is not guaranteed to produce JSON and nothing but JSON. For a more deterministic response, you can pass a specific JSON schema in a [`responseSchema`](https://ai.google.dev/api/rest/v1beta/GenerationConfig#FIELDS.response_schema) field so that Gemini always responds with an expected structure.

This guide shows you how to generate JSON using the [`generateContent`](https://ai.google.dev/api/rest/v1/models/generateContent) method through the SDK of your choice or using the REST API directly. The examples show text-only input, although Gemini can also produce JSON responses to multimodal requests that include [images](https://ai.google.dev/gemini-api/docs/vision), [videos](https://ai.google.dev/gemini-api/docs/vision), and [audio](https://ai.google.dev/gemini-api/docs/audio).

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
    

Generate JSON
-------------

When the model is configured to output JSON, it responds to any prompt with JSON-formatted output.

You can control the structure of the JSON response by supplying a schema. There are two ways to supply a schema to the model:

*   As text in the prompt
*   As a structured schema supplied through model configuration

Both approaches work in both Gemini 1.5 Flash and Gemini 1.5 Pro.

### Supply a schema as text in the prompt

The following example prompts the model to return cookie recipes in a specific JSON format.

Since the model gets the format specification from text in the prompt, you may have some flexibility in how you represent the specification. Any reasonable format for representing a JSON schema may work.

```
import google.generativeai as genai

model = genai.GenerativeModel("gemini-1.5-pro-latest")
prompt = """List a few popular cookie recipes in JSON format.

Use this JSON schema:

Recipe = {'recipe_name': str, 'ingredients': list[str]}
Return: list[Recipe]"""
result = model.generate_content(prompt)
print(result)controlled_generation.py
```
The output might look like this:

\[{"ingredients": \["1 cup (2 sticks) unsalted butter, softened", "1 cup granulated sugar", "1 cup packed light brown sugar", "2 teaspoons pure vanilla extract", "2 large eggs", "3 cups all-purpose flour", "1 teaspoon baking soda", "1 teaspoon salt", "2 cups chocolate chips"\], "recipe\_name": "Chocolate Chip Cookies"}, {"ingredients": \["1 cup (2 sticks) unsalted butter, softened", "¾ cup granulated sugar", "¾ cup packed light brown sugar", "1 teaspoon pure vanilla extract", "2 large eggs", "2 ¼ cups all-purpose flour", "1 teaspoon baking soda", "1 teaspoon salt", "1 cup rolled oats", "1 cup raisins"\], "recipe\_name": "Oatmeal Raisin Cookies"}, {"ingredients": \["1 cup (2 sticks) unsalted butter, softened", "1 ½ cups powdered sugar", "1 teaspoon pure vanilla extract", "2 ¼ cups all-purpose flour", "¼ teaspoon salt", "Sprinkles or colored sugar for decoration"\], "recipe\_name": "Sugar Cookies"}, {"ingredients": \["1 cup peanut butter", "1 cup granulated sugar", "1 large egg"\], "recipe\_name": "3-Ingredient Peanut Butter Cookies"}\]

### Supply a schema through model configuration

The following example does the following:

1.  Instantiates a model configured through a schema to respond with JSON.
2.  Prompts the model to return cookie recipes.

```
import google.generativeai as genai

import typing_extensions as typing

class Recipe(typing.TypedDict):
    recipe_name: str
    ingredients: list[str]

model = genai.GenerativeModel("gemini-1.5-pro-latest")
result = model.generate_content(
    "List a few popular cookie recipes.",
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json", response_schema=list[Recipe]
    ),
)
print(result)controlled_generation.py
```
The output might look like this:

\[{"ingredients": \["1 cup (2 sticks) unsalted butter, softened", "1 cup granulated sugar", "1 cup packed light brown sugar", "2 teaspoons pure vanilla extract", "2 large eggs", "3 cups all-purpose flour", "1 teaspoon baking soda", "1 teaspoon salt", "2 cups chocolate chips"\], "recipe\_name": "Chocolate Chip Cookies"}, {"ingredients": \["1 cup (2 sticks) unsalted butter, softened", "¾ cup granulated sugar", "¾ cup packed light brown sugar", "1 teaspoon pure vanilla extract", "2 large eggs", "2 ¼ cups all-purpose flour", "1 teaspoon baking soda", "1 teaspoon salt", "1 cup rolled oats", "1 cup raisins"\], "recipe\_name": "Oatmeal Raisin Cookies"}, {"ingredients": \["1 cup (2 sticks) unsalted butter, softened", "1 ½ cups powdered sugar", "1 teaspoon pure vanilla extract", "2 ¼ cups all-purpose flour", "¼ teaspoon salt", "Sprinkles or colored sugar for decoration"\], "recipe\_name": "Sugar Cookies"}, {"ingredients": \["1 cup peanut butter", "1 cup granulated sugar", "1 large egg"\], "recipe\_name": "3-Ingredient Peanut Butter Cookies"}\]

#### Schema Definition Syntax

Specify the schema for the JSON response in the `response_schema` property of your model configuration. The value of `response_schema` must be a either:

*   A type hint annotation, as defined in the Python [`typing` module](https://docs.python.org/3/library/typing.html) module.
*   An instance of [`genai.protos.Schema`](https://ai.google.dev/api/python/google/generativeai/protos/Schema).

##### Define a Schema with a Type Hint Annotation

The easiest way to define a schema is with a type hint annotation. This is the approach used in the preceding example:

```
generation_config={"response_mime_type": "application/json",
                   "response_schema": list[Recipe]}
```
The Gemini API Python client library supports schemas defined with the following subset of `typing` annotations (where `AllowedType` is any allowed type annotation):

*   `int`
*   `float`
*   `bool`
*   `str` (or enum)
*   `list[AllowedType]`
*   For dict types:
    *   `dict[str, AllowedType]`. This annotation declares all dict values to be the same type, but doesn't specify what keys should be included.
    *   User-defined subclasses of [`typing.TypedDict`](https://docs.python.org/3/library/typing.html#typing.TypedDict). This approach lets you specify the key names and define different types for the values associated with each of the keys.
    *   User-defined [Data Classes](https://docs.python.org/3/library/dataclasses.html). Like `TypedDict` subclasses, this approach lets you specify the key names and define different types for the values associated with each of the keys.

##### Define a Schema with `genai.protos.Schema` Protocol Buffer

The Gemini API `genai.protos.Schema` protocol buffer definition supports a few additional schema features not supported for type hints, including:

*   Enums for strings
*   Specifying the format for numeric types (`int32` or `int64` for integers, for example)
*   Specifying which fields are required.

If you need these features, instantiate a `genai.protos.Schema` using one of the methods illustrated in [Function Calling: Low Level Access](https://ai.google.dev/gemini-api/docs/function-calling/tutorial?lang=python#optional_low_level_access).

Use an enum to constrain output
-------------------------------

In some cases you might want the model to choose a single option from a list of options. To implement this behavior, you can pass an _enum_ in your schema. You can use an enum option anywhere you could use a `str` in the `response_schema`, because an enum is actually a list of strings. Like a JSON schema, an enum lets you constrain model output to meet the requirements of your application.

For example, assume that you're developing an application to classify images of musical instruments into one of five categories: `"Percussion"`, `"String"`, `"Woodwind"`, `"Brass"`, or "`Keyboard`". You could create an enum to help with this task.

Before running the code examples in this section, make sure to import the Google Generative AI library:

```
import google.generativeai as genai
```
In the following example, you pass the enum class `Choice` as the `response_schema`, and the model should choose the most appropriate enum option.

```
import google.generativeai as genai

import enum

class Choice(enum.Enum):
    PERCUSSION = "Percussion"
    STRING = "String"
    WOODWIND = "Woodwind"
    BRASS = "Brass"
    KEYBOARD = "Keyboard"

model = genai.GenerativeModel("gemini-1.5-pro-latest")

organ = genai.upload_file(media / "organ.jpg")
result = model.generate_content(
    ["What kind of instrument is this:", organ],
    generation_config=genai.GenerationConfig(
        response_mime_type="text/x.enum", response_schema=Choice
    ),
)
print(result)  # Keyboardcontrolled_generation.py
```
The Python SDK will translate the type declarations for the API. But the API actually accepts a subset of the OpenAPI 3.0 schema ([Schema](https://ai.google.dev/api/caching#schema)). You can also pass the schema as JSON:

```
import google.generativeai as genai

model = genai.GenerativeModel("gemini-1.5-pro-latest")

organ = genai.upload_file(media / "organ.jpg")
result = model.generate_content(
    ["What kind of instrument is this:", organ],
    generation_config=genai.GenerationConfig(
        response_mime_type="text/x.enum",
        response_schema={
            "type": "STRING",
            "enum": ["Percussion", "String", "Woodwind", "Brass", "Keyboard"],
        },
    ),
)
print(result)  # Keyboardcontrolled_generation.py
```
Beyond basic multiple choice problems, you can use an enum anywhere in a schema for JSON or function calling. For example, you could ask the model for a list of recipe titles and use a `Grade` enum to give each title a popularity grade:

```
import google.generativeai as genai

import enum
from typing_extensions import TypedDict

class Grade(enum.Enum):
    A_PLUS = "a+"
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    F = "f"

class Recipe(TypedDict):
    recipe_name: str
    grade: Grade

model = genai.GenerativeModel("gemini-1.5-pro-latest")

result = model.generate_content(
    "List about 10 cookie recipes, grade them based on popularity",
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json", response_schema=list[Recipe]
    ),
)
print(result)  # [{"grade": "a+", "recipe_name": "Chocolate Chip Cookies"}, ...]controlled_generation.py
```
To get started with enums, try the [enum quickstart Colab](https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Enum.ipynb).