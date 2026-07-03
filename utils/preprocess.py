"""Fungsi bantu untuk preprocessing gambar sebelum prediksi."""

import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)


def preprocess_image(image_path_or_file):
    """Buka gambar, resize ke 224x224, normalisasi ke [0,1], tambah batch dim."""
    img = Image.open(image_path_or_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr
