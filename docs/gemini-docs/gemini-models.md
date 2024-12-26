Title: Gemini models

URL Source: https://ai.google.dev/gemini-api/docs/models/gemini

Markdown Content:
2.0 Flash

Our newest multimodal model, with next generation features and improved capabilities

*   Input audio, images, video, and text — get text, image, and audio responses
*   Features low-latency conversational interactions with our Multimodal Live API

1.5 Flash

Our most balanced multimodal model with great performance for most tasks

*   Input audio, images, video, and text, get text responses
*   Generate code, extract data, edit text, and more
*   Best for tasks balancing performance and cost

1.5 Pro

Our best performing multimodal model with features for a wide variety of reasoning tasks

*   Input audio, images, video, and text, get text responses
*   Generate code, extract data, edit text, and more
*   For when you need a boost in performance

Model variants
--------------

The Gemini API offers different models that are optimized for specific use cases. Here's a brief overview of Gemini variants that are available:

| Model variant | Input(s) | Output | Optimized for |
| --- | --- | --- | --- |
| [Gemini 2.0 Flash](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-2.0-flash)  
`gemini-2.0-flash-exp` | Audio, images, videos, and text | Text, images (coming soon), and audio (coming soon) | Next generation features, speed, and multimodal generation for a diverse variety of tasks |
| [Gemini 1.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-flash)  
`gemini-1.5-flash` | Audio, images, videos, and text | Text | Fast and versatile performance across a diverse variety of tasks |
| [Gemini 1.5 Flash-8B](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-flash-8b)  
`gemini-1.5-flash-8b` | Audio, images, videos, and text | Text | High volume and lower intelligence tasks |
| [Gemini 1.5 Pro](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-pro)  
`gemini-1.5-pro` | Audio, images, videos, and text | Text | Complex reasoning tasks requiring more intelligence |
| [Gemini 1.0 Pro](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.0-pro)  
`gemini-1.0-pro`  
(Deprecated on 2/15/2025) | Text | Text | Natural language tasks, multi-turn text and code chat, and code generation |
| [Text Embedding](https://ai.google.dev/gemini-api/docs/models/gemini#text-embedding-and-embedding)  
`text-embedding-004` | Text | Text embeddings | Measuring the relatedness of text strings |
| [AQA](https://ai.google.dev/gemini-api/docs/models/gemini#AQA)  
`aqa` | Text | Text | Providing source-grounded answers to questions |

### Gemini 2.0 Flash **(Experimental)**

Gemini 2.0 Flash delivers next-gen features and improved capabilities, including superior speed, native tool use, multimodal generation, and a 1M token context window. Learn more about Gemini 2.0 Flash in our [overview page](https://ai.google.dev/gemini-api/docs/models/gemini-v2).

[Try in Google AI Studio](https://aistudio.google.com/?model=gemini-2.0-flash-exp)

#### Model details

| Property | Description |
| --- | --- |
| Model code | `models/gemini-2.0-flash-exp` |
| Supported data types | 
**Inputs**

Audio, images, video, and text

**Output**

Audio (coming soon), images (coming soon), and text



 |
| Token limits[\[\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#token-size) | 

**Input token limit**

1,048,576

**Output token limit**

8,192



 |
| Rate limits[\[\*\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#rate-limits) | 

*   10 RPM
*   4 million TPM
*   1,500 RPD



 |
| Capabilities | 

**Structured outputs**

Supported

**Caching**

Not supported

**Tuning**

Not supported

**Function calling**

Supported

**Code execution**

Supported

**Search**

Supported

**Image generation**

Supported

**Native tool use**

Supported

**Audio generation**

Supported

**Multimodal Live API**

Supported



 |
| Versions | 

Read the [model version patterns](https://ai.google.dev/gemini-api/docs/models/gemini#model-versions) for more details.

*   Latest: `gemini-2.0-flash-exp`



 |
| Latest update | December 2024 |
| Knowledge cutoff | August 2024 |

### Gemini 1.5 Flash

Gemini 1.5 Flash is a fast and versatile multimodal model for scaling across diverse tasks.

[Try in Google AI Studio](https://aistudio.google.com/?model=gemini-1.5-flash)

#### Model details

| Property | Description |
| --- | --- |
| Model code | `models/gemini-1.5-flash` |
| Supported data types | 
**Inputs**

Audio, images, video, and text

**Output**

Text



 |
| Token limits[\[\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#token-size) | 

**Input token limit**

1,048,576

**Output token limit**

8,192



 |
| Audio/visual specs | 

**Maximum number of images per prompt**

3,600

**Maximum video length**

1 hour

**Maximum audio length**

Approximately 9.5 hours



 |
| Rate limits[\[\*\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#rate-limits) | 

**Free:**

*   15 RPM
*   1 million TPM
*   1,500 RPD

**Pay-as-you-go:**

*   2,000 RPM
*   4 million TPM



 |
| Capabilities | 

**System instructions**

Supported

**JSON mode**

Supported

**JSON schema**

Supported

**Adjustable safety settings**

Supported

**Caching**

Supported

**Tuning**

Supported

**Function calling**

Supported

**Code execution**

Supported

**Bidirectional streaming**

Not supported



 |
| Versions | 

Read the [model version patterns](https://ai.google.dev/gemini-api/docs/models/gemini#model-versions) for more details.

*   Latest: `gemini-1.5-flash-latest`
*   Latest stable: `gemini-1.5-flash`
*   Stable:

*   `gemini-1.5-flash-001`
*   `gemini-1.5-flash-002`





 |
| Latest update | September 2024 |

### Gemini 1.5 Flash-8B

Gemini 1.5 Flash-8B is a small model designed for lower intelligence tasks.

[Try in Google AI Studio](https://aistudio.google.com/?model=gemini-1.5-flash)

#### Model details

| Property | Description |
| --- | --- |
| Model code | `models/gemini-1.5-flash-8b` |
| Supported data types | 
**Inputs**

Audio, images, video, and text

**Output**

Text



 |
| Token limits[\[\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#token-size) | 

**Input token limit**

1,048,576

**Output token limit**

8,192



 |
| Audio/visual specs | 

**Maximum number of images per prompt**

3,600

**Maximum video length**

1 hour

**Maximum audio length**

Approximately 9.5 hours



 |
| Rate limits[\[\*\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#rate-limits) | 

**Free:**

*   15 RPM
*   1 million TPM
*   1,500 RPD

**Pay-as-you-go:**

*   4,000 RPM
*   4 million TPM



 |
| Capabilities | 

**System instructions**

Supported

**JSON mode**

Supported

**JSON schema**

Supported

**Adjustable safety settings**

Supported

**Caching**

Supported

**Tuning**

Supported

**Function calling**

Supported

**Code execution**

Supported

**Bidirectional streaming**

Not supported



 |
| Versions | 

Read the [model version patterns](https://ai.google.dev/gemini-api/docs/models/gemini#model-versions) for more details.

*   Latest: `gemini-1.5-flash-8b-latest`
*   Latest stable: `gemini-1.5-flash-8b`
*   Stable:

*   `gemini-1.5-flash-8b-001`





 |
| Latest update | October 2024 |

### Gemini 1.5 Pro

Gemini 1.5 Pro is a mid-size multimodal model that is optimized for a wide-range of reasoning tasks. 1.5 Pro can process large amounts of data at once, including 2 hours of video, 19 hours of audio, codebases with 60,000 lines of code, or 2,000 pages of text.

[Try in Google AI Studio](https://aistudio.google.com/?model=gemini-1.5-pro)

#### Model details

| Property | Description |
| --- | --- |
| Model code | `models/gemini-1.5-pro` |
| Supported data types | 
**Inputs**

Audio, images, video, and text

**Output**

Text



 |
| Token limits[\[\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#token-size) | 

**Input token limit**

2,097,152

**Output token limit**

8,192



 |
| Audio/visual specs | 

**Maximum number of images per prompt**

7,200

**Maximum video length**

2 hours

**Maximum audio length**

Approximately 19 hours



 |
| Rate limits[\[\*\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#rate-limits) | 

**Free:**

*   2 RPM
*   32,000 TPM
*   50 RPD

**Pay-as-you-go:**

*   1,000 RPM
*   4 million TPM



 |
| Capabilities | 

**System instructions**

Supported

**JSON mode**

Supported

**JSON schema**

Supported

**Adjustable safety settings**

Supported

**Caching**

Supported

**Tuning**

Not supported

**Function calling**

Supported

**Code execution**

Supported

**Bidirectional streaming**

Not supported



 |
| Versions | 

Read the [model version patterns](https://ai.google.dev/gemini-api/docs/models/gemini#model-versions) for more details.

*   Latest: `gemini-1.5-pro-latest`
*   Latest stable: `gemini-1.5-pro`
*   Stable:

*   `gemini-1.5-pro-001`
*   `gemini-1.5-pro-002`





 |
| Latest update | September 2024 |

### Gemini 1.0 Pro **(Deprecated)**

Gemini 1.0 Pro is an NLP model that handles tasks like multi-turn text and code chat, and code generation.

[Try in Google AI Studio](https://aistudio.google.com/?model=gemini-1.0-pro)

#### Model details

| Property | Description |
| --- | --- |
| Model code | `models/gemini-1.0-pro` |
| Supported data types | 
**Input**

Text

**Output**

Text



 |
| Rate limits[\[\*\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#rate-limits) | 

**Free:**

*   15 RPM
*   32,000 TPM
*   1,500 RPD

**Pay-as-you-go:**

*   360 RPM
*   120,000 TPM
*   30,000 RPD



 |
| Capabilities | 

**System instructions**

Not supported

**JSON mode**

Not supported

**JSON schema**

Not supported

**Adjustable safety settings**

Supported

**Caching**

Not supported

**Tuning**

Supported

**Function calling**

Supported

**Function calling configuration**

Not supported

**Code execution**

Not supported

**Bidirectional streaming**

Not supported



 |
| Versions | 

*   Latest: `gemini-1.0-pro-latest`
*   Latest stable: `gemini-1.0-pro`
*   Stable: `gemini-1.0-pro-001`



 |
| Latest update | February 2024 |

### Text Embedding and Embedding

#### Text Embedding

[Text embeddings](https://ai.google.dev/gemini-api/docs/embeddings) are used to measure the relatedness of strings and are widely used in many AI applications.

`text-embedding-004` achieves a [stronger retrieval performance and outperforms existing models](https://arxiv.org/pdf/2403.20327) with comparable dimensions, on the standard MTEB embedding benchmarks.

##### Model details

| Property | Description |
| --- | --- |
| Model code | 
**Gemini API**

`models/text-embedding-004`



 |
| Supported data types | 

**Input**

Text

**Output**

Text embeddings



 |
| Token limits[\[\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#token-size) | 

**Input token limit**

2,048

**Output dimension size**

768



 |
| Rate limits[\[\*\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#rate-limits) | 1,500 requests per minute |
| Adjustable safety settings | Not supported |
| Latest update | April 2024 |

#### Embedding

You can use the Embedding model to generate [text embeddings](https://ai.google.dev/gemini-api/docs/embeddings) for input text.

The Embedding model is optimized for creating embeddings with 768 dimensions for text of up to 2,048 tokens.

##### Embedding model details

| Property | Description |
| --- | --- |
| Model code | `models/embedding-001` |
| Supported data types | 
**Input**

Text

**Output**

Text embeddings



 |
| Token limits[\[\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#token-size) | 

**Input token limit**

2,048

**Output dimension size**

768



 |
| Rate limits[\[\*\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#rate-limits) | 1,500 requests per minute |
| Adjustable safety settings | Not supported |
| Latest update | December 2023 |

### AQA

You can use the AQA model to perform [Attributed Question-Answering](https://ai.google.dev/gemini-api/docs/semantic_retrieval) (AQA)–related tasks over a document, corpus, or a set of passages. The AQA model returns answers to questions that are grounded in provided sources, along with estimating answerable probability.

#### Model details

| Property | Description |
| --- | --- |
| Model code | `models/aqa` |
| Supported data types | 
**Input**

Text

**Output**

Text



 |
| Supported language | English |
| Token limits[\[\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#token-size) | 

**Input token limit**

7,168

**Output token limit**

1,024



 |
| Rate limits[\[\*\*\]](https://ai.google.dev/gemini-api/docs/models/gemini#rate-limits) | 1,500 requests per minute |
| Adjustable safety settings | Supported |
| Latest update | December 2023 |

See the [examples](https://ai.google.dev/examples) to explore the capabilities of these model variations.

\[\*\] A token is equivalent to about 4 characters for Gemini models. 100 tokens are about 60-80 English words.

\[\*\*\] _RPM: Requests per minute_  
_TPM: Tokens per minute_  
_RPD: Requests per day_  
_TPD: Tokens per day_

Due to capacity limitations, specified maximum rate limits are not guaranteed.

Model version name patterns
---------------------------

Gemini models are available in either _preview_ or _stable_ versions. In your code, you can use one of the following model name formats to specify which model and version you want to use.

*   **Latest:** Points to the cutting-edge version of the model for a specified generation and variation. The underlying model is updated regularly and might be a preview version. Only exploratory testing apps and prototypes should use this alias.
    
    To specify the latest version, use the following pattern: `<model>-<generation>-<variation>-latest`. For example, `gemini-1.0-pro-latest`.
    
*   **Latest stable:** Points to the most recent stable version released for the specified model generation and variation.
    
    To specify the latest stable version, use the following pattern: `<model>-<generation>-<variation>`. For example, `gemini-1.0-pro`.
    
*   **Stable:** Points to a specific stable model. Stable models don't change. Most production apps should use a specific stable model.
    
    To specify a stable version, use the following pattern: `<model>-<generation>-<variation>-<version>`. For example, `gemini-1.0-pro-001`.
    
*   **Experimental:** Points to an experimental model available in Preview, as defined in the [Terms](https://ai.google.dev/gemini-api/terms), meaning it is not for production use. We release experimental models to gather feedback, get our latest updates into the hands of developers quickly, and highlight the pace of innovation happening at Google. What we learn from experimental launches informs how we release models more widely. An experimental model can be swapped for another without prior notice. We don't guarantee that an experimental model will become a stable model in the future.
    
    To specify an experimental version, use the following pattern: `<model>-<generation>-<variation>-<version>`. For example, `gemini-exp-1121`.
    

Available languages
-------------------

Gemini models are trained to work with the following languages:

*   Arabic (`ar`)
*   Bengali (`bn`)
*   Bulgarian (`bg`)
*   Chinese simplified and traditional (`zh`)
*   Croatian (`hr`)
*   Czech (`cs`)
*   Danish (`da`)
*   Dutch (`nl`)
*   English (`en`)
*   Estonian (`et`)
*   Finnish (`fi`)
*   French (`fr`)
*   German (`de`)
*   Greek (`el`)
*   Hebrew (`iw`)
*   Hindi (`hi`)
*   Hungarian (`hu`)
*   Indonesian (`id`)
*   Italian (`it`)
*   Japanese (`ja`)
*   Korean (`ko`)
*   Latvian (`lv`)
*   Lithuanian (`lt`)
*   Norwegian (`no`)
*   Polish (`pl`)
*   Portuguese (`pt`)
*   Romanian (`ro`)
*   Russian (`ru`)
*   Serbian (`sr`)
*   Slovak (`sk`)
*   Slovenian (`sl`)
*   Spanish (`es`)
*   Swahili (`sw`)
*   Swedish (`sv`)
*   Thai (`th`)
*   Turkish (`tr`)
*   Ukrainian (`uk`)
*   Vietnamese (`vi`)