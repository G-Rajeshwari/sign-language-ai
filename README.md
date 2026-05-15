# Sign Language to Text & Speech AI

Converts ASL hand signs to text and speech in real-time using MediaPipe and Random Forest.

## Setup
1. Install Python 3.11
2. Create virtual environment: `py -3.11 -m venv venv`
3. Activate: `venv\Scripts\activate`
4. Install libraries: `python -m pip install mediapipe==0.10.9 opencv-python scikit-learn pyttsx3 numpy`

## Usage
1. Download ASL dataset from Kaggle and place in `data/` folder
2. `python 2_extract_landmarks.py` — extract hand landmarks
3. `python 3_train_model.py` — train the model (98.9% accuracy)
4. `python 4_run_detector.py` — run live detector

## Tech Stack
- MediaPipe — hand landmark detection
- OpenCV — webcam feed
- Scikit-learn — Random Forest classifier
- pyttsx3 — text to speech