"""
train.py
Training model klasifikasi kesegaran buah menggunakan Transfer Learning VGG16.

Struktur dataset yang diharapkan:
dataset/
    train/
        fresh_apple/
        rotten_apple/
        fresh_banana/
        rotten_banana/
        ...
    (opsional) val/  -> jika tidak ada, akan dibuat otomatis dari train via validation_split

Jalankan:
    python train.py
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset/train"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.0001
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# DATA GENERATOR + AUGMENTATION
# ---------------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)

with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f, indent=2)

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(train_generator.class_indices, f)

print(f"Jumlah kelas: {num_classes} -> {class_names}")

# ---------------------------------------------------------------------------
# MODEL: VGG16 TRANSFER LEARNING
# ---------------------------------------------------------------------------
base_model = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base_model.trainable = False  # freeze seluruh convolution layer

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint(
        os.path.join(MODEL_DIR, "model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]

# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Simpan model final (format .keras dan .h5)
model.save(os.path.join(MODEL_DIR, "model.keras"))
model.save(os.path.join(MODEL_DIR, "model.h5"))

# ---------------------------------------------------------------------------
# GRAFIK ACCURACY & LOSS
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_history.png"))
plt.close()

# ---------------------------------------------------------------------------
# EVALUASI: CONFUSION MATRIX & CLASSIFICATION REPORT
# ---------------------------------------------------------------------------
val_generator.reset()
y_pred_prob = model.predict(val_generator)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = val_generator.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
plt.close()

report = classification_report(y_true, y_pred, target_names=class_names)
print(report)
with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

final_val_acc = max(history.history["val_accuracy"])
final_val_loss = min(history.history["val_loss"])

with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
    json.dump(
        {
            "num_classes": num_classes,
            "class_names": class_names,
            "input_shape": [224, 224, 3],
            "val_accuracy": float(final_val_acc),
            "val_loss": float(final_val_loss),
            "total_params": int(model.count_params()),
        },
        f,
        indent=2,
    )

print("Training selesai. Model dan laporan tersimpan di folder 'model/'.")
