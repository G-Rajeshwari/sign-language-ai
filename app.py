import streamlit as st
import pickle
import numpy as np
import cv2
import os
import av
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🤟")
st.title("🤟 Hand Gesture Recognition")
st.markdown("Show an ASL hand sign to your webcam — AI detects it live!")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    return pickle.load(open(model_path, 'rb'))

@st.cache_resource
def load_hand_detector():
    task_path = "/tmp/hand_landmarker.task"
    if not os.path.exists(task_path):
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, task_path)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1
    )
    return vision.HandLandmarker.create_from_options(options)

model = load_model()

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

    try:
        detector = load_hand_detector()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            coords = [v for p in lm for v in (p.x, p.y)]

            # Draw landmarks manually
            h, w = img.shape[:2]
            for point in lm:
                cx, cy = int(point.x * w), int(point.y * h)
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

            # Draw connections
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),
                (5,9),(9,13),(13,17)
            ]
            for a, b in connections:
                ax = int(lm[a].x * w); ay = int(lm[a].y * h)
                bx = int(lm[b].x * w); by = int(lm[b].y * h)
                cv2.line(img, (ax, ay), (bx, by), (0, 255, 0), 2)

            if len(coords) == 42:
                pred = model.predict([coords])[0]
                cv2.rectangle(img, (0, 0), (420, 70), (0, 0, 0), -1)
                cv2.putText(img, f'Sign: {pred}', (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
        else:
            cv2.putText(img, 'No hand detected', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    except Exception as e:
        cv2.putText(img, 'Loading...', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.markdown("---")
st.subheader("📷 Live Webcam Detection")
st.info("👇 Click START and allow camera access. Hold hand sign steady!")

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
st.caption("Hand Gesture Recognition | MediaPipe Tasks + Random Forest | Streamlit")