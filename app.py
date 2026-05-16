import streamlit as st
import pickle
import numpy as np
import cv2
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🤟")
st.title("🤟 Hand Gesture Recognition")
st.markdown("Show an ASL hand sign to your webcam — the AI will recognize it live!")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    return pickle.load(open(model_path, 'rb'))

model = load_model()

IMG_SIZE = 32  # must match training size

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class HandGestureProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        features = gray.flatten() / 255.0
        pred = model.predict([features])[0]
        cv2.putText(img, f'Sign: {pred}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        return frame.from_ndarray(img, format="bgr24")

st.markdown("---")
st.subheader("📷 Live Webcam Detection")
st.info("Click START and allow camera access when browser asks.")

webrtc_streamer(
    key="hand-gesture",
    video_processor_factory=HandGestureProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
st.caption("Hand Gesture Recognition | ASL Alphabet | Built with Streamlit")