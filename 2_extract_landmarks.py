import mediapipe as mp
import pickle, cv2, os

# Updated import for mediapipe 0.10+
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Use the legacy solutions via this path
mp_hands_module = mp.solutions.hands
mp_hands = mp_hands_module.Hands(static_image_mode=True, max_num_hands=1)

data, labels = [], []
LIMIT = 300  # images per class

for label in os.listdir('./data'):
    folder = f'./data/{label}'
    if not os.path.isdir(folder):
        continue
    files = os.listdir(folder)[:LIMIT]
    count = 0
    for img_file in files:
        img = cv2.imread(f'{folder}/{img_file}')
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(img_rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            coords = [v for p in lm.landmark for v in (p.x, p.y)]
            data.append(coords)
            labels.append(label)
            count += 1

    print(f"✅ {label}: {count} samples processed")

pickle.dump({'data': data, 'labels': labels}, open('dataset.pkl', 'wb'))
print(f"\n✅ Total saved: {len(data)} samples to dataset.pkl")