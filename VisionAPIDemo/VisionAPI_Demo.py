import os, io
from google.cloud import vision
from google.cloud.vision import types
import pandas as pd
import json

def detectText(img):
    with io.open(img,'rb') as image_file:
        content = image_file.read()
    
    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    df = pd.DataFrame(columns=['locale','description'])
    for text in texts:
        df = df.append(
            dict(
                locale=text.locale,
                description=text.description
            ),
            ignore_index=True
        )
    return df['description'][0]

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'/media/tuminh14/New Volume/Final Project/FE_CREDIT_FINAL/VisionAPIDemo/ServiceAccountToken.json' #de path file json cua may nha!!!!!

client = vision.ImageAnnotatorClient()

FILE_NAME = 'tung_cmnd.jpg'
FOLDER_PATH = r'/media/tuminh14/New Volume/Final Project/Data/CMND/'
print(detectText(os.path.join(FOLDER_PATH, FILE_NAME)))