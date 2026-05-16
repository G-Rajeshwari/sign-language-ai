import streamlit as st
import mediapipe as mp
import pickle
import cv2
import numpy as np
from PIL import Image

# Load model
model = pickle.load(open('model.pkl', 'rb'))
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

st.title("🤟 Sign Language to Text AI")
st.write("Show an ASL hand sign to your webcam and the AI will recognize it!")

sentence = st.session_state.get("sentence", "")

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])
text_display = st.empty()

cap = cv2.VideoCapture(0)

prev_char = ""
hold_count = 0

while run:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        coords = [v for p in lm.landmark for v in (p.x, p.y)]
        pred = model.predict([coords])[0]

        if pred == prev_char:
            hold_count += 1
        else:
            hold_count = 0
            prev_char = pred

        if hold_count == 10:
            sentence += pred + " "
            st.session_state["sentence"] = sentence
            hold_count = 0

        cv2.putText(frame, f'Sign: {pred}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.putText(frame, sentence[-40:], (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    text_display.markdown(f"### 📝 Sentence: `{sentence}`")

cap.release()

if st.button("🗑️ Clear Sentence"):
    st.session_state["sentence"] = ""