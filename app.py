
import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model

st.title("🌱 Farmer AI Plant Disease Detector")

model = load_model("plant_disease_model.keras", compile=False)

with open("class_names.json") as f:
    class_names = json.load(f)

uploaded_file = st.file_uploader("Upload Leaf Image")

if uploaded_file:

    img = Image.open(uploaded_file)

    st.image(img)

    img = img.resize((224,224))
    img = np.array(img)/255
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)

    index = np.argmax(pred)

    disease = class_names[index]

    st.success("Detected Disease: " + disease)
