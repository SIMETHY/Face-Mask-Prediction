import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_my_model():
    return load_model("face_mask_model.h5")

model = load_my_model()

# Class labels
class_names = ["With Mask", "Without Mask"]

# -------------------------------
# App UI
# -------------------------------
st.title("😷 Face Mask Detection App")
st.write("Upload an image to check whether the person is wearing a mask or not.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # -------------------------------
    # Preprocess Image
    # -------------------------------
    img_resized = img.resize((128, 128))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # -------------------------------
    # Prediction
    # -------------------------------
    prediction = model.predict(img_array)

    # Debug (optional)
    st.write("Raw Prediction:", prediction)

    # Handle binary vs multi-class
    prediction = model.predict(img_array)

    prob = prediction[0][0]   # single output

    if prob < 0.5:
        pred_class = 0   # With Mask
        confidence = (1 - prob) * 100
    else:
        pred_class = 1   # Without Mask
        confidence = prob * 100

    # -------------------------------
    # Output
    # -------------------------------
    st.subheader("Prediction:")
    st.write(f"**{class_names[pred_class]}**")
    st.write(f"Confidence: {confidence:.2f}%")

    if pred_class == 0:
        st.success("✅ Person is wearing a mask")
    else:
        st.error("❌ Person is NOT wearing a mask")
