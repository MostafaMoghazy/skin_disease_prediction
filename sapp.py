import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

# Load doctor data
# doctors_df = pd.read_csv("doctors.csv")

# Set title
st.title("🧴 Skin Disease Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload an image of the skin area", type=["jpg", "jpeg", "png"])

# Define prediction and preprocessing
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_disease(image_array):
    preds = model.predict(image_array)
    class_index = np.argmax(preds)
    confidence = np.max(preds)
    return class_index, confidence

# Class names (update with your model's classes)
class_names = ['eczema', 'psoriasis', 'melanoma', 'acne']  # Example

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    image_array = preprocess_image(image)
    class_index, confidence = predict_disease(image_array)
    disease = class_names[class_index]

    st.markdown(f"### 🧬 Prediction: *{disease.capitalize()}* with {confidence*100:.2f}% confidence")

    # # Suggest doctor
    # st.subheader("👨‍⚕ Suggested Doctor")
    # doctor_match = doctors_df[doctors_df["disease_treated"].str.lower() == disease.lower()]
    # if not doctor_match.empty:
    #     doc = doctor_match.sample(1).iloc[0]
    #     st.markdown(f"- *Name*: {doc['name']}")
    #     st.markdown(f"- *Specialization*: {doc['specialization']}")
    #     st.markdown(f"- *Contact*: {doc['contact']}")
    # else:
    #     st.warning("Sorry, no matching doctor found for this disease.")
