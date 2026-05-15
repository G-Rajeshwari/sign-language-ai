import cv2, pickle, pyttsx3
import mediapipe as mp

model = pickle.load(open('model.pkl', 'rb'))

engine = pyttsx3.init()
engine.setProperty('rate', 150)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
sentence = ""
prev_char = ""
hold_count = 0

print("✅ Detector running — show hand signs to camera!")
print("   Press Q to quit | Press C to clear sentence")

while True:
    ret, frame = cap.read()
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
            engine.say(pred)
            engine.runAndWait()
            hold_count = 0

        cv2.putText(frame, f'Sign: {pred}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.putText(frame, sentence[-40:], (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, 'Q=Quit  C=Clear', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Sign Language AI', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        sentence = ""

cap.release()
cv2.destroyAllWindows()