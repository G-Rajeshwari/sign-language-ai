import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

d = pickle.load(open('dataset.pkl', 'rb'))
X, y = d['data'], d['labels']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"✅ Model Accuracy: {acc * 100:.1f}%")

pickle.dump(model, open('model.pkl', 'wb'))
print("✅ Model saved to model.pkl")