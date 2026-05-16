import streamlit as st
import mediapipe as mp
import pickle
import cv2
import numpy as np
import os

st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🤟")
st.title("🤟 Hand Gesture Recognition")
st.write("Upload an image of an ASL hand sign and the AI will recognize it!")

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

# New mediapipe API (works on all versions)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download hand landmarker model
import urllib.request
model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
task_path = "/tmp/hand_landmarker.task"
if not os.path.exists(task_path):
    with st.spinner("Downloading hand detection model..."):
        urllib.request.urlretrieve(model_url, task_path)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=task_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

st.markdown("---")
st.subheader("📸 Upload a Hand Sign Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    with HandLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        coords = [v for p in lm for v in (p.x, p.y)]

        if len(coords) == 42:
            pred = model.predict([coords])[0]
            st.success(f"### ✅ Detected Sign: `{pred}`")
            st.balloons()
        else:
            st.warning("Could not extract enough landmarks. Try a clearer image.")
    else:
        st.warning("No hand detected in the image. Please upload a clear hand sign photo.")

st.markdown("---")
st.caption("Built with MediaPipe + Random Forest | ASL Alphabet Recognition")