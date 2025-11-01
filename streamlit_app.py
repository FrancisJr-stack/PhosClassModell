import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

#Load the model (once, at app start)
MODEL_PATH = "Ripe_Unripe_classification.keras"
model = load_model(MODEL_PATH)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.90

#Streamlit UI
st.set_page_config(page_title="Tomato Classifier", page_icon="🍅", layout="centered")
st.title("🍅 Ripe Tomato Classifier")
st.markdown(
    """
     Upload a **tomato** image (ripe or unripe).
     CODE WITH FRANCIS JUNIOR
    """
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction logic
if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    #Pre-process
    img_resized = image.resize((64, 64))                 # match training size
    img_array   = img_to_array(img_resized)
    img_array   = np.expand_dims(img_array, axis=0) / 255.0   # normalize if needed

    # Predict
    probs = model.predict(img_array, verbose=0)[0]        # shape = (2,)

    # Handle different output shapes
    if probs.shape[0] == 2:               # softmax → [ripe_prob, unripe_prob]
        Unripe_prob, Ripe_prob = probs[0], probs[1]
    elif probs.shape[0] == 1:             # sigmoid → single value
        Unripe_prob = probs[0]
        Ripe_prob = 1 - Unripe_prob
    else:
        st.error("Unexpected model output shape.")
        st.stop()

    # Confidence check
    max_prob = max(Unripe_prob, Ripe_prob)

    # **Both** probs < threshold → treat as *unknown* (not a tomato)
    if Unripe_prob < CONFIDENCE_THRESHOLD and Ripe_prob < CONFIDENCE_THRESHOLD:
        st.error("🚫 **Unknown image** – this does not look like a ripe or unripe tomato.")
    else:
        # Normal classification (one class is confident)
        if max_prob < CONFIDENCE_THRESHOLD:
            st.warning(
                f"Low confidence ({max_prob:.2%}). The image may be blurry or ambiguous."
            )
        # Class mapping: 0 = Ripe, 1 = Unripe
        if Unripe_prob > Ripe_prob:               # Ripe wins
            st.success("🍅 **This is an Unripe Tomato.** Please wait for it to ripen.")
        else:                                     # Unripe wins
            st.info("🍅 **This is a Ripe Tomato!** Yummy! 😋")

        #Show probabilities
        st.write(f"**UnRipe** probability: {Unripe_prob:.2%}")
        st.write(f"**ripe** probability: {Ripe_prob:.2%}")


# Optional styling
st.markdown(
    """
    <style>
    body {background-color: #f0f4f8; font-family: 'Arial', sans-serif;}
    </style>
    """,
    unsafe_allow_html=True,
)
