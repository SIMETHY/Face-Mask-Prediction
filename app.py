import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# -------------------------------
# Custom CSS (LIGHT MODE)
# -------------------------------
st.markdown("""
<style>

/* Background */
body {
    background-color: #f5f9ff;
}

/* Main container spacing */
.block-container {
    padding-top: 2rem;
}

/* Image styling */
img {
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.05);
}

/* Footer */
.footer {
    text-align: center;
    color: #6c757d;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown("""
<div style="
    background: linear-gradient(90deg, #007BFF, #00C6FF);
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    color: white;
    font-size: 30px;
    font-weight: 600;
    box-shadow: 0px 6px 20px rgba(0,123,255,0.25);
">
    😷 Face Mask Detection
    <div style="font-size:15px; font-weight:400; margin-top:5px;">
        Upload an image to detect mask usage
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_my_model():
    return load_model("face_mask_model.h5")

model = load_my_model()

class_names = ["With Mask", "Without Mask"]

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📌 About")
st.sidebar.info("""
This app uses a CNN model to detect whether a person is wearing a mask or not.

Steps:
1. Upload image
2. Model processes it
3. Shows prediction + confidence
""")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    # -------------------------------
    # Show Image
    # -------------------------------
    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

    # -------------------------------
    # Preprocess
    # -------------------------------
    img_resized = img.resize((128, 128))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # -------------------------------
    # Prediction
    # -------------------------------
    with st.spinner("🧠 Analyzing Image..."):
        prediction = model.predict(img_array)

    prob = prediction[0][0]

    if prob < 0.5:
        pred_class = 0
        confidence = (1 - prob) * 100
        color = "#28a745"   # green
        message = "✅ Wearing Mask"
    else:
        pred_class = 1
        confidence = prob * 100
        color = "#dc3545"   # red
        message = "❌ No Mask"

    # -------------------------------
    # Results
    # -------------------------------
    with col2:

        # Result Card
        st.markdown(f"""
        <div style="
            background: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            font-size: 24px;
            font-weight: 600;
            color: {color};
            border: 2px solid {color};
            box-shadow: 0px 6px 20px rgba(0,0,0,0.05);
        ">
            {message}
        </div>
        """, unsafe_allow_html=True)

        # Confidence Bar
        st.markdown("### 📊 Confidence Level")

        st.markdown(f"""
        <div style="
            background-color: #eaf3ff;
            border-radius: 10px;
            padding: 5px;
        ">
            <div style="
                width: {confidence}%;
                background: linear-gradient(90deg, #007BFF, #00C6FF);
                padding: 10px;
                border-radius: 10px;
                text-align: right;
                color: white;
                font-weight: 600;
            ">
                {confidence:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write(f"**Prediction:** {class_names[pred_class]}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Confidence badge
        if confidence > 90:
            st.success("🔥 High Confidence")
        elif confidence > 70:
            st.info("👍 Moderate Confidence")
        else:
            st.warning("⚠️ Low Confidence")

    # -------------------------------
    # Debug Toggle
    # -------------------------------
    show_debug = st.checkbox("🔍 Show Raw Prediction")
    if show_debug:
        st.write(prediction)

