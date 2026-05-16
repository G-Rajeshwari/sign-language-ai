import os
import cv2
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = './data'
IMG_SIZE = 64
LIMIT = 200

data, labels = [], []

for label in os.listdir(DATA_DIR):
    folder = f'{DATA_DIR}/{label}'
    if not os.path.isdir(folder):
        continue
    files = os.listdir(folder)[:LIMIT]
    count = 0
    for img_file in files:
        img = cv2.imread(f'{folder}/{img_file}')
        if img is None:
            continue
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        features = gray.flatten() / 255.0
        data.append(features)
        labels.append(label)
        count += 1
    print(f"✅ {label}: {count} samples")

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, random_state=42)

print("\n🔄 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Accuracy: {acc * 100:.1f}%")

pickle.dump(model, open('model.pkl', 'wb'))
print("✅ model.pkl saved!")