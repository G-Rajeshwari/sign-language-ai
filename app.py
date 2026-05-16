import streamlit as st
import pickle
import numpy as np
import cv2
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🤟")
st.title("🤟 Hand Gesture Recognition")
st.markdown("Show an ASL hand sign to your webcam — the AI will recognize it live!")

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

IMG_SIZE = 64

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class HandGestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.prediction = "Waiting..."

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocess
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        features = gray.flatten() / 255.0

        # Predict
        pred = model.predict([features])[0]
        self.prediction = pred

        # Draw on frame
        cv2.putText(img, f'Sign: {pred}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(img, 'Hand Gesture Recognition', (10, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return frame.from_ndarray(img, format="bgr24")

st.markdown("---")
st.subheader("📷 Live Webcam Detection")
st.info("Allow camera access when your browser asks. Hold a hand sign steady for best results.")

ctx = webrtc_streamer(
    key="hand-gesture",
    video_processor_factory=HandGestureProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:
    st.markdown(f"### 🔤 Detected: `{ctx.video_processor.prediction}`")

st.markdown("---")
st.caption("Built with Streamlit WebRTC | ASL Alphabet Recognition | Random Forest AI")