import streamlit as st
import pandas as pd
from pathlib import Path
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
from PIL import Image

validate_ = "Ripe_Unripe_classification.keras"
model = load_model(validate_)
# streamlit app

st.title("Juniors App")
st.write("PLEASE INPUT A SINGLE IMAGE OF RIPE OR UNRIPE TOMATOES")
st.write("Ripe and Unripe Tomatoes Classification")
uploaded = st.file_uploader("choose an image", type=['jpg', 'png', 'jpeg'])
if uploaded:
    image = Image.open(uploaded)
    image = image.resize((64, 64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    result = model.predict(image)
    for i in result:
        # val = model.predict(uploaded)
        if i == 1:
            st.write("An unripe tomato")
            st.write("Please do not eat")
        else:
            st.write("This is a ripe tomato")
            st.write("Yummy")

#     model.summary()
