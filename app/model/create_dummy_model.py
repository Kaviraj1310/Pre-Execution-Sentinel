# app/model/create_dummy_model.py
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

MODEL_DIR = os.path.dirname(__file__)
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "malware_cnn_model.h5")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

def build_and_save_dummy_model():
    # Simple CNN that matches preprocess output shape (128,128,1)
    model = Sequential([
        Conv2D(8, (3,3), activation='relu', input_shape=(128,128,1)),
        MaxPooling2D(2,2),
        Conv2D(16, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    # Create tiny random dataset just to "fit" briefly (optional)
    X = np.random.rand(20, 128, 128, 1).astype('float32')
    y = np.random.randint(0, 2, size=(20, 1))

    # Quick fit to initialise weights (one epoch)
    model.fit(X, y, epochs=1, batch_size=4, verbose=0)

    # Save real HDF5 model
    model.save(MODEL_PATH)
    print(f"✅ Dummy model saved to: {MODEL_PATH} (size: {os.path.getsize(MODEL_PATH)} bytes)")

    # Save label encoder (very simple)
    label_encoder = {"classes": ["benign", "malware"]}
    joblib.dump(label_encoder, ENCODER_PATH)
    print(f"✅ Dummy label encoder saved to: {ENCODER_PATH}")

if __name__ == "__main__":
    build_and_save_dummy_model()
    print("Dummy model and encoder creation complete.")
    