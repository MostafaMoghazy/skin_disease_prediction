import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from bs4 import BeautifulSoup
import requests
import pandas as pd


def load_disease_model(path="skindisease.keras"):
    return load_model(path)


def predict_disease(model, img_file, class_names):
    img = image.load_img(img_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence


def scrape_doctors(city, specialty="dermatology"):
    url = f"https://www.vezeeta.com/en/doctor/{specialty.lower()}/{city.lower()}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.content, "lxml")
    names = soup.find_all(
        'a', {'class': 'CommonStylesstyle__TransparentA-sc-1vkcu2o-2 cTFrlk'})
    specs = soup.find_all(
        'p', {'class': 'DoctorCardSubComponentsstyle__Text-sc-1vq3h7c-14 DoctorCardSubComponentsstyle__DescText-sc-1vq3h7c-17'})
    locs = soup.find_all(
        'span', {'class': 'DoctorCardstyle__Text-sc-uptab2-4'})

    data = []
    for i in range(min(len(names), len(specs), len(locs))):
        try:
            data.append({
                "name": names[i].text.strip(),
                "specialty": specs[i].text.strip(),
                "location": locs[i].text.strip()
            })
        except Exception:
            continue

    return pd.DataFrame(data)

