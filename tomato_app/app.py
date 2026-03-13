import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model("tomato_pune_model.h5")

IMG_SIZE = 224
classes = [
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato_Target_Spot',
 'Tomato_Tomato_YellowLeaf_Curl_Virus',
 'Tomato_Tomato_mosaic_virus',
 'Tomato_healthy'
]
# Pune rule engine
def pune_recommendation(disease):
    if disease == "Tomato_healthy":
        return "Healthy. Avoid over-irrigation in Vertisol. Use raised beds."

    if "blight" in disease.lower():
        return "High fungal risk in Pune monsoon. Use Mancozeb. Improve drainage."

    if "mite" in disease.lower():
        return "Use neem oil spray. Avoid water stress."

    if "virus" in disease.lower():
        return "Remove infected plants immediately. Control whiteflies."

    return "Apply organic treatment and monitor."
# Prediction
def predict(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    class_index = np.argmax(pred, axis=1)[0]
    return classes[class_index]

# UI
st.title("Tomato Disease Detection - Pune AI")
st.write("Upload tomato leaf image")

file = st.file_uploader("Choose image", type=["jpg","png","jpeg"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    disease = predict(image)
    advice = pune_recommendation(disease)

    st.subheader("Prediction:")
    st.success(disease)

    st.subheader("Recommendation:")
    st.info(advice)
