# run.py
import os
import numpy as np
from tensorflow.keras import layers, models
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "app", "model")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "malware_cnn.h5")
LABEL_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

# --- Synthetic dataset (DEMO ONLY) ---
# Replace the following with real feature arrays and labels for production.
num_samples = 600
X = np.random.rand(num_samples, 128, 128, 1).astype("float32")
y = np.random.choice([0, 1], size=(num_samples,))  # 1 = malware, 0 = benign

# Build simple CNN
def build_model():
    model = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=(128,128,1)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Training demo model on synthetic data (this is a placeholder).")
    model = build_model()
    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.15)
    model.save(MODEL_PATH)
    # label encoder saved as mapping dict for demo
    joblib.dump({"classes": ["benign", "malware"]}, LABEL_PATH)
    print(f"Saved demo model to {MODEL_PATH}")
    print(f"Saved label metadata to {LABEL_PATH}")