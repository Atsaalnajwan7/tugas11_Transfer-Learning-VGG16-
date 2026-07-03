"""
app.py
Aplikasi web Flask untuk klasifikasi kesegaran buah menggunakan model
Transfer Learning VGG16.
"""

import os
import json
import uuid

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np

from utils.preprocess import preprocess_image

app = Flask(__name__)
app.secret_key = "fruit-freshness-secret-key"
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

MODEL_DIR = "model"
MODEL_PATH_KERAS = os.path.join(MODEL_DIR, "model.keras")
MODEL_PATH_H5 = os.path.join(MODEL_DIR, "model.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")

model = None
class_names = []
model_info = {}


def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            return json.load(f)
    return []


def load_model_info():
    if os.path.exists(MODEL_INFO_PATH):
        with open(MODEL_INFO_PATH) as f:
            return json.load(f)
    return {}


def load_model():
    """Lazy-load model Keras. Mengembalikan None jika model belum tersedia."""
    global model
    if model is not None:
        return model
    try:
        import tensorflow as tf

        if os.path.exists(MODEL_PATH_KERAS):
            model = tf.keras.models.load_model(MODEL_PATH_KERAS)
        elif os.path.exists(MODEL_PATH_H5):
            model = tf.keras.models.load_model(MODEL_PATH_H5)
        else:
            model = None
    except Exception as e:
        print(f"Gagal memuat model: {e}")
        model = None
    return model


class_names = load_class_names()
model_info = load_model_info()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def parse_label(label):
    """Ubah 'apple_fresh' -> ('Apple', 'Fresh')"""
    parts = label.split("_")
    status = parts[-1].capitalize()
    fruit = " ".join(p.capitalize() for p in parts[:-1])
    return fruit, status


@app.route("/")
def index():
    return render_template("index.html", model_ready=os.path.exists(MODEL_PATH_KERAS) or os.path.exists(MODEL_PATH_H5))


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/dataset")
def dataset():
    stats = []
    dataset_dir = os.path.join("dataset", "train")
    total = 0
    if os.path.isdir(dataset_dir):
        for cls in sorted(os.listdir(dataset_dir)):
            cls_dir = os.path.join(dataset_dir, cls)
            if os.path.isdir(cls_dir):
                n = len([f for f in os.listdir(cls_dir) if allowed_file(f)])
                stats.append({"name": cls, "count": n})
                total += n
    return render_template("dataset.html", stats=stats, total=total)


@app.route("/model")
def model_page():
    info = load_model_info()
    return render_template("model.html", info=info, class_names=class_names)


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("Tidak ada file yang diunggah.")
        return redirect(url_for("predict_page"))

    file = request.files["file"]
    if file.filename == "":
        flash("Silakan pilih gambar terlebih dahulu.")
        return redirect(url_for("predict_page"))

    if not allowed_file(file.filename):
        flash("Format file tidak didukung. Gunakan JPG, JPEG, PNG, atau WEBP.")
        return redirect(url_for("predict_page"))

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    m = load_model()
    if m is None:
        flash("Model belum tersedia. Jalankan 'python train.py' terlebih dahulu.")
        return redirect(url_for("predict_page"))

    try:
        arr = preprocess_image(save_path)
    except Exception:
        # File corrupt / tidak bisa dibaca sebagai gambar
        if os.path.exists(save_path):
            os.remove(save_path)
        flash("File gambar tidak dapat dibaca (kemungkinan rusak atau format tidak didukung). Coba unggah gambar lain.")
        return redirect(url_for("predict_page"))

    preds = m.predict(arr)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100

    labels = class_names if class_names else [f"class_{i}" for i in range(len(preds))]
    fruit, status = parse_label(labels[top_idx])

    probabilities = [
        {"label": labels[i], "fruit": parse_label(labels[i])[0], "status": parse_label(labels[i])[1], "value": round(float(p) * 100, 2)}
        for i, p in enumerate(preds)
    ]
    probabilities.sort(key=lambda x: x["value"], reverse=True)

    low_confidence = confidence < 70

    return render_template(
        "result.html",
        image_url=url_for("static", filename=f"uploads/{unique_name}"),
        fruit=fruit,
        status=status,
        confidence=round(confidence, 2),
        probabilities=probabilities,
        low_confidence=low_confidence,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)