# 🤟 Hand Gesture Recognition AI

A real-time ASL (American Sign Language) hand gesture recognition system built with MediaPipe, Random Forest, and Streamlit.

## 🌐 Live Demo
 [handgesturerecognition.streamlit.app](https://handgesturerecognition.streamlit.app)

##  What it does
- Opens your webcam in the browser
- Detects your hand in real-time using MediaPipe
- Draws skeleton joints and connections on your hand
- Predicts the ASL letter you're showing (A–Z + space, delete, nothing)
- Displays the result live on screen

## 🧠 How it works
1. **MediaPipe Tasks API** detects 21 hand landmarks (x, y coordinates)
2. **42 features** (x,y of each landmark) are extracted per frame
3. **Random Forest Classifier** (trained on Kaggle ASL dataset) predicts the sign
4. Result is overlaid on the live webcam feed via **Streamlit WebRTC**

## 🗂️ Project Structure

```
sign_language_ai/
├── app.py                     # Streamlit web app (live webcam detection)
├── 2_extract_landmarks.py     # Extract MediaPipe landmarks from dataset
├── 3_train_model.py           # Train Random Forest classifier
├── 5_retrain_pixel_model.py   # Alternative pixel-based model
├── 6_small_model.py           # Lightweight model for fast loading
├── model.pkl                  # Trained classifier (98.9% accuracy)
├── requirements.txt           # Python dependencies
├── packages.txt               # System dependencies
├── runtime.txt                # Python version (3.11)
├── .python-version            # Python version pin
└── .gitignore
```
## ⚙️ Tech Stack
| Tool | Purpose |
|------|---------|
| MediaPipe Tasks API | Hand landmark detection |
| OpenCV | Video frame processing |
| scikit-learn | Random Forest classifier |
| Streamlit | Web interface |
| streamlit-webrtc | Browser webcam access |
| aiortc | WebRTC protocol |

## 📊 Model Performance
- **Dataset:** Kaggle ASL Alphabet (87,000 images, 29 classes)
- **Training samples:** 7,348 landmark sets
- **Accuracy:** 98.9%
- **Classes:** A–Z + space, delete, nothing

## 🚀 Run Locally

### Prerequisites
- Python 3.11
- Webcam

### Setup
```bash
# Clone the repo
git clone https://github.com/G-Rajeshwari/sign-language-ai.git
cd sign-language-ai

# Create virtual environment
py -3.11 -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
python -m pip install mediapipe==0.10.9 opencv-python scikit-learn pyttsx3 numpy streamlit streamlit-webrtc aiortc

# Run the web app
streamlit run app.py

# OR run the local desktop version
python 4_run_detector.py
```

## 📁 Dataset
Dataset used: [ASL Alphabet - Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## 👩‍💻 Author
**G. Rajeshwari**  
GitHub: [@G-Rajeshwari](https://github.com/G-Rajeshwari)