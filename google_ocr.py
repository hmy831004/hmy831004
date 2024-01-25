import os
import io
# Imports the Google Cloud client library
from google.cloud import vision
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'nia-chart-summary-056b9142b638.json' 

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.abspath('./data/type2/img2_4.jpg')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

# For dense text, use document_text_detection
# For less dense text, use text_detection
response = client.document_text_detection(image=image)
texts = response.full_text_annotation
print("Detected text: {}".format(texts.text))

response = client.document_text_detection(image=image)
texts = response.text_annotations

# print(' '.join([text.description for i,text in enumerate(texts)]))

# Performs label detection on the image file
# for page in texts.pages:
#     for block in page.blocks:
#         print('\nBlock confidence: {}\n'.format(block.confidence))
