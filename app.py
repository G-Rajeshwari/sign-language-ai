import streamlit as st
import pickle
import numpy as np
import cv2
import os
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🤟")
st.title("🤟 Hand Gesture Recognition")
st.markdown("Show an ASL hand sign to your webcam — the AI will recognize it live!")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    return pickle.load(open(model_path, 'rb'))

model = load_model()
IMG_SIZE = 32

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]}
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Preprocess
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    features = gray.flatten() / 255.0

    # Predict
    pred = model.predict([features])[0]

    # Draw black background box for text
    cv2.rectangle(img, (0, 0), (400, 70), (0, 0, 0), -1)

    # Draw prediction text
    cv2.putText(img, f'Sign: {pred}', (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)

    # Draw border
    cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1),
                  (0, 255, 0), 3)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown("---")
st.subheader("📷 Live Webcam Detection")
st.info("👇 Click START below and allow camera access when browser asks.")

webrtc_streamer(
    key="hand-gesture",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
st.markdown("**Signs supported:** A B C D E F G H I J K L M N O P Q R S T U V W X Y Z + space, delete, nothing")
st.caption("Hand Gesture Recognition | ASL Alphabet | Built with Streamlit WebRTC")