from array import array
import os
import sys
import time
from os import listdir
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import pickle
from tqdm import tqdm

# Add your Computer Vision subscription key to your environment variables.
if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
    subscription_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']
else:
    print("\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()
# Add your Computer Vision endpoint to your environment variables.
if 'COMPUTER_VISION_ENDPOINT' in os.environ:
    endpoint = os.environ['COMPUTER_VISION_ENDPOINT']
else:
    print("\nSet the COMPUTER_VISION_ENDPOINT environment variable.\n**Restart your shell or IDE for changes to take effect.**")
    sys.exit()

analyze_url = endpoint + "vision/v2.1/analyze"


def main():
    #computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    dir = './results/vm/landscapes-no-cond/generated_imgs1000/'

    existing_analyses = listdir(dir + 'analysis')

    for i, filename in tqdm(enumerate(listdir(dir), start=0)):
        if(filename.endswith('png')):
            analysis_filename = filename[:-4] + '.p'
            if(analysis_filename in existing_analyses):
                continue

            analysis = get_analysis(dir + filename)
            pickle.dump(analysis,open(dir + 'analysis/' +filename[:-4] + '.p','wb'))
            if i % 15 ==0:
                time.sleep(60) # Free tier only allows a certain number of calls per minute
    


def get_analysis(image_path):
    # Read the image into a byte array
    image_data = open(image_path, "rb").read()
    headers = {'Ocp-Apim-Subscription-Key': subscription_key,
               'Content-Type': 'application/octet-stream'}
    params = {'visualFeatures': 'Categories,Description,Color'}
    response = requests.post(
        analyze_url, headers=headers, params=params, data=image_data)
    response.raise_for_status()

    # The 'analysis' object contains various fields that describe the image. The most
    # relevant caption for the image is obtained from the 'description' property.
    analysis = response.json()
    print(analysis)
    # image_caption = analysis["description"]["captions"][0]["text"].capitalize()

    # Display the image and overlay it with the caption.
    # image = Image.open(BytesIO(image_data))
    # plt.imshow(image)
    # plt.axis("off")
    # _ = plt.title(image_caption, size="x-large", y=-0.1)
    # plt.show()

    return analysis

if __name__ == "__main__":
    main()