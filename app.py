import streamlit as st
import pickle
import numpy as np
import cv2
import os
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🤟")
st.title("🤟 Hand Gesture Recognition")
st.markdown("Show an ASL hand sign to your webcam — AI detects it live with skeleton overlay!")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    return pickle.load(open(model_path, 'rb'))

# Load the landmark-based model (42 features)
model = load_model()

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]}
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:
        results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]

        # Draw skeleton
        mp_draw.draw_landmarks(
            img, lm,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )

        # Extract features
        coords = [v for p in lm.landmark for v in (p.x, p.y)]

        if len(coords) == 42:
            pred = model.predict([coords])[0]

            # Black box + green text
            cv2.rectangle(img, (0, 0), (420, 70), (0, 0, 0), -1)
            cv2.putText(img, f'Sign: {pred}', (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
    else:
        cv2.putText(img, 'No hand detected', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown("---")
st.subheader("📷 Live Webcam Detection")
st.info("👇 Click START and allow camera access. Hold your hand sign steady!")

webrtc_streamer(
    key="hand-gesture",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
st.markdown("**Signs:** A–Z + space, delete, nothing")
st.caption("Hand Gesture Recognition | MediaPipe + Random Forest | Streamlit WebRTC")