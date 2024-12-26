Title: Explore vision capabilities with the Gemini API

URL Source: https://ai.google.dev/gemini-api/docs/vision?lang=python

Markdown Content:
The Gemini API is able to process images and videos, enabling a multitude of exciting developer use cases. Some of Gemini's vision capabilities include the ability to:

*   Caption and answer questions about images
*   Transcribe and reason over PDFs, including long documents up to 2 million token context window
*   Describe, segment, and extract information from videos, including both visual frames and audio, up to 90 minutes long
*   Detect objects in an image and return bounding box coordinates for them

This tutorial demonstrates some possible ways to prompt the Gemini API with images and video input, provides code examples, and outlines prompting best practices with multimodal vision capabilities. All output is text-only.

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
    

Prompting with images
---------------------

In this tutorial, you will upload images using the File API or as inline data and generate content based on those images.

### Technical details (images)

Gemini 1.5 Pro and 1.5 Flash support a maximum of 3,600 image files.

Images must be in one of the following image data MIME types:

*   PNG - `image/png`
*   JPEG - `image/jpeg`
*   WEBP - `image/webp`
*   HEIC - `image/heic`
*   HEIF - `image/heif`

Each image is equivalent to 258 tokens.

While there are no specific limits to the number of pixels in an image besides the model's context window, larger images are scaled down to a maximum resolution of 3072x3072 while preserving their original aspect ratio, while smaller images are scaled up to 768x768 pixels. There is no cost reduction for images at lower sizes, other than bandwidth, or performance improvement for images at higher resolution.

For best results:

*   Rotate images to the correct orientation before uploading.
*   Avoid blurry images.
*   If using a single image, place the text prompt after the image.

Image input
-----------

For total image payload size less than 20MB, we recommend either uploading base64 encoded images or directly uploading locally stored image files.

### Base64 encoded images

You can upload public image URLs by encoding them as Base64 payloads. We recommend using the httpx library to fetch the image URLs. The following code example shows how to do this:

```
import httpx
import os
import base64

model = genai.GenerativeModel(model_name = "gemini-1.5-pro")
image_path = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall.jpg/2560px-Palace_of_Westminster_from_the_dome_on_Methodist_Central_Hall.jpg"

image = httpx.get(image_path)

prompt = "Caption this image."
response = model.generate_content([{'mime_type':'image/jpeg', 'data': base64.b64encode(image.content).decode('utf-8')}, prompt])

print(response.text)
```

### Multiple images

To prompt with multiple images in Base64 encoded format, you can do the following:

```
import httpx
import os
import base64

model = genai.GenerativeModel(model_name = "gemini-1.5-pro")
image_path_1 = "path/to/your/image1.jpeg"  # Replace with the actual path to your first image
image_path_2 = "path/to/your/image2.jpeg" # Replace with the actual path to your second image

image_1 = httpx.get(image_path_1)
image_2 = httpx.get(image_path_2)

prompt = "Generate a list of all the objects contained in both images."

response = model.generate_content([
{'mime_type':'image/jpeg', 'data': base64.b64encode(image_1.content).decode('utf-8')},
{'mime_type':'image/jpeg', 'data': base64.b64encode(image_2.content).decode('utf-8')}, prompt])

print(response.text)
```

### Upload one or more locally stored image files

Alternatively, you can upload one or more locally stored image files.

```
import PIL.Image
import os
import google.generativeai as genai

image_path_1 = "path/to/your/image1.jpeg"  # Replace with the actual path to your first image
image_path_2 = "path/to/your/image2.jpeg" # Replace with the actual path to your second image

sample_file_1 = PIL.Image.open(image_path_1)
sample_file_2 = PIL.Image.open(image_path_2)

#Choose a Gemini model.
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

prompt = "Write an advertising jingle based on the items in both images."

response = model.generate_content([prompt, sample_file_1, sample_file_2])

print(response.text)
```
Note that these inline data calls don't include many of the features available through the File API, such as getting file metadata, [listing](https://ai.google.dev/gemini-api/docs/vision?lang=python#list-files), or [deleting files](https://ai.google.dev/gemini-api/docs/vision?lang=python#delete-files).

### Large image payloads

When the combination of files and system instructions that you intend to send is larger than 20 MB in size, use the File API to upload those files.

Use the [`media.upload`](https://ai.google.dev/api/rest/v1beta/media/upload) method of the File API to upload an image of any size.

After uploading the file, you can make `GenerateContent` requests that reference the File API URI. Select the generative model and provide it with a text prompt and the uploaded image.

```
import google.generativeai as genai

myfile = genai.upload_file(media / "Cajun_instruments.jpg")
print(f"{myfile=}")

model = genai.GenerativeModel("gemini-1.5-flash")
result = model.generate_content(
    [myfile, "\n\n", "Can you tell me about the instruments in this photo?"]
)
print(f"{result.text=}")files.py
```

### OpenAI Compatibility

You can access Gemini's image understanding capabilities using the OpenAI libraries. This lets you integrate Gemini into existing OpenAI workflows by updating three lines of code and using your Gemini API key. See the [Image understanding example](https://ai.google.dev/gemini-api/docs/openai#image-understanding) for code demonstrating how to send images encoded as Base64 payloads.

Capabilities
------------

This section outlines specific vision capabilities of the Gemini model, including object detection and bounding box coordinates.

### Get a bounding box for an object

Gemini models are trained to return bounding box coordinates as relative widths or heights in the range of \[0, 1\]. These values are then scaled by 1000 and converted to integers. Effectively, the coordinates represent the bounding box on a 1000x1000 pixel version of the image. Therefore, you'll need to convert these coordinates back to the dimensions of your original image to accurately map the bounding boxes.

```
# Choose a Gemini model.
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

prompt = "Return a bounding box for each of the objects in this image in [ymin, xmin, ymax, xmax] format."
response = model.generate_content([sample_file_1, prompt])

print(response.text)
```
The model returns bounding box coordinates in the format `[ymin, xmin, ymax, xmax]`. To convert these normalized coordinates to the pixel coordinates of your original image, follow these steps:

1.  Divide each output coordinate by 1000.
2.  Multiply the x-coordinates by the original image width.
3.  Multiply the y-coordinates by the original image height.

To explore more detailed examples of generating bounding box coordinates and visualizing them on images, we encourage you to review our [Object Detection cookbook example](https://github.com/google-gemini/cookbook/blob/main/examples/Object_detection.ipynb).

Prompting with video
--------------------

In this tutorial, you will upload a video using the File API and generate content based on those images.

### Technical details (video)

Gemini 1.5 Pro and Flash support up to approximately an hour of video data.

Video must be in one of the following video format MIME types:

*   `video/mp4`
*   `video/mpeg`
*   `video/mov`
*   `video/avi`
*   `video/x-flv`
*   `video/mpg`
*   `video/webm`
*   `video/wmv`
*   `video/3gpp`

The File API service extracts image frames from videos at 1 frame per second (FPS) and audio at 1Kbps, single channel, adding timestamps every second. These rates are subject to change in the future for improvements in inference.

Individual frames are 258 tokens, and audio is 32 tokens per second. With metadata, each second of video becomes ~300 tokens, which means a 1M context window can fit slightly less than an hour of video.

To ask questions about time-stamped locations, use the format `MM:SS`, where the first two digits represent minutes and the last two digits represent seconds.

For best results:

*   Use one video per prompt.
*   If using a single video, place the text prompt after the video.

### Upload a video file using the File API

The File API accepts video file formats directly. This example uses the short NASA film ["Jupiter's Great Red Spot Shrinks and Grows"](https://www.youtube.com/watch?v=JDi4IdtvDVE0). Credit: Goddard Space Flight Center (GSFC)/David Ladd (2018).

"Jupiter's Great Red Spot Shrinks and Grows" is in the public domain and does not show identifiable people. ([NASA image and media usage guidelines.](https://www.nasa.gov/nasa-brand-center/images-and-media/))

Start by retrieving the short video:

```
wget https://storage.googleapis.com/generativeai-downloads/images/GreatRedSpot.mp4
```
Upload the video using the File API and print the URI.

```
# Upload the video and print a confirmation.
video_file_name = "GreatRedSpot.mp4"

print(f"Uploading file...")
video_file = genai.upload_file(path=video_file_name)
print(f"Completed upload: {video_file.uri}")
```

### Verify file upload and check state

Verify the API has successfully received the files by calling the [`files.get`](https://ai.google.dev/api/rest/v1beta/files/get) method.

```
import time

# Check whether the file is ready to be used.
while video_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(10)
    video_file = genai.get_file(video_file.name)

if video_file.state.name == "FAILED":
  raise ValueError(video_file.state.name)

```

### Prompt with a video and text

Once the uploaded video is in the `ACTIVE` state, you can make `GenerateContent` requests that specify the File API URI for that video. Select the generative model and provide it with the uploaded video and a text prompt.

```
# Create the prompt.
prompt = "Summarize this video. Then create a quiz with answer key based on the information in the video."

# Choose a Gemini model.
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Make the LLM request.
print("Making LLM inference request...")
response = model.generate_content([video_file, prompt],
                                  request_options={"timeout": 600})

# Print the response, rendering any Markdown
Markdown(response.text)
```

### Refer to timestamps in the content

You can use timestamps of the form `HH:MM:SS` to refer to specific moments in the video.

```
# Create the prompt.
prompt = "What are the examples given at 01:05 and 01:19 supposed to show us?"

# Choose a Gemini model.
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Make the LLM request.
print("Making LLM inference request...")
response = model.generate_content([video_file, prompt],
                                  request_options={"timeout": 600})
print(response.text)
```

### Transcribe video and provide visual descriptions

The Gemini models can transcribe and provide visual descriptions of video content by processing both the audio track and visual frames. For visual descriptions, the model samples the video at a rate of **1 frame per second**. This sampling rate may affect the level of detail in the descriptions, particularly for videos with rapidly changing visuals.

```
# Create the prompt.
prompt = "Transcribe the audio from this video, giving timestamps for salient events in the video. Also provide visual descriptions."

# Choose a Gemini model.
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Make the LLM request.
print("Making LLM inference request...")
response = model.generate_content([video_file, prompt],
                                  request_options={"timeout": 600})
print(response.text)
```

List files
----------

You can list all files uploaded using the File API and their URIs using [`files.list`](https://ai.google.dev/api/files#method:-files.list).

```
import google.generativeai as genai

print("My files:")
for f in genai.list_files():
    print("  ", f.name)files.py
```

Delete files
------------

Files uploaded using the File API are automatically deleted after 2 days. You can also manually delete them using [`files.delete`](https://ai.google.dev/api/files#method:-files.delete).

```
import google.generativeai as genai

myfile = genai.upload_file(media / "poem.txt")

myfile.delete()

try:
    # Error.
    model = genai.GenerativeModel("gemini-1.5-flash")
    result = model.generate_content([myfile, "Describe this file."])
except google.api_core.exceptions.PermissionDenied:
    passfiles.py
```

What's next
-----------

This guide shows how to upload image and video files using the File API and then generate text outputs from image and video inputs. To learn more, see the following resources:

*   [File prompting strategies](https://ai.google.dev/gemini-api/docs/file-prompting-strategies): The Gemini API supports prompting with text, image, audio, and video data, also known as multimodal prompting.
*   [System instructions](https://ai.google.dev/gemini-api/docs/system-instructions): System instructions let you steer the behavior of the model based on your specific needs and use cases.
*   [Safety guidance](https://ai.google.dev/gemini-api/docs/safety-guidance): Sometimes generative AI models produce unexpected outputs, such as outputs that are inaccurate, biased, or offensive. Post-processing and human evaluation are essential to limit the risk of harm from such outputs.