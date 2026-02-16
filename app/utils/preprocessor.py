import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

TARGET_SIZE = (128, 128)

def bytes_to_image(byte_data: bytes, target_size: tuple = TARGET_SIZE) -> np.ndarray:
    if not byte_data:
        raise ValueError("Byte data cannot be empty.")
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)
    length = byte_array.size
    side = int(np.ceil(np.sqrt(length)))
    padding_needed = side * side - length
    padded = np.pad(byte_array, (0, padding_needed), 'constant', constant_values=0)
    img2d = padded.reshape((side, side))
    img = Image.fromarray(img2d.astype('uint8'), mode='L')
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = img_to_array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,1)
    return arr

def preprocess_upload(file_path: str):
    try:
        with open(file_path, "rb") as f:
            b = f.read()
        return bytes_to_image(b)
    except Exception as e:
        print("Preprocess error:", e)
        return None
