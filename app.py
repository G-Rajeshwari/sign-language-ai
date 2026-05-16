import streamlit as st
import pickle
import numpy as np
import cv2
import os
from PIL import Image

st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🤟")
st.title("🤟 Hand Gesture Recognition")
st.markdown("Upload a clear image of an ASL hand sign — the AI will identify it!")

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

try:
    model = pickle.load(open(model_path, 'rb'))
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

st.markdown("---")
st.subheader("📸 Upload a Hand Sign Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img = np.array(pil_img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing hand sign..."):
        # Resize to 64x64 and flatten as features
        resized = cv2.resize(img_bgr, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        features = gray.flatten() / 255.0

        try:
            pred = model.predict([features])[0]
            st.success(f"### ✅ Detected Sign: `{pred}`")
            st.balloons()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("The model was trained with MediaPipe landmarks (42 features). It needs to be retrained for image-based input.")

st.markdown("---")
st.caption("Hand Gesture Recognition | ASL Alphabet | Built with Streamlit")