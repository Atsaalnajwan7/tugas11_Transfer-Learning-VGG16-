# Fruit Freshness Classifier (VGG16 + Flask) — Simple Version

Klasifikasi kesegaran buah (Fresh/Rotten) untuk Apple, Banana, dan Strawberry
menggunakan Transfer Learning VGG16, dengan antarmuka web Flask + Bootstrap 5.

## Dataset

Dataset sudah disertakan di `dataset/train/` (hasil pembersihan dari
`Fruit_Freshness_Dataset.zip`, duplikat format gambar sudah dihapus dan
semua gambar dikonversi ke `.jpg`):

| Kelas              | Jumlah |
|---------------------|--------|
| apple_fresh          | 109 |
| apple_rotten         | 52  |
| banana_fresh         | 16  |
| banana_rotten        | 90  |
| strawberry_fresh     | 209 |
| strawberry_rotten    | 44  |

> Catatan: kelas `banana_fresh` cukup sedikit (16 gambar). Untuk hasil lebih
> baik, tambahkan lebih banyak contoh gambar pisang segar bila memungkinkan.

## Menjalankan Project

```bash
pip install -r requirements.txt

# 1. Latih model (hasil disimpan di folder model/)
python train.py

# 2. Jalankan web app
python app.py
```

Buka `http://localhost:5000` di browser.

## Struktur Project

```
FruitFreshnessVGG16/
├── app.py                # Flask app
├── train.py               # Script training VGG16 transfer learning
├── requirements.txt
├── runtime.txt
├── vercel.json
├── model/                 # Model hasil training (model.keras, model.h5, dll)
├── dataset/train/          # Dataset (6 kelas)
├── static/                # CSS, JS, uploads
├── templates/              # Halaman HTML (Bootstrap 5)
└── utils/preprocess.py     # Preprocessing gambar untuk prediksi
```

## Fitur Web

- **Home** — pengantar project
- **About** — penjelasan Transfer Learning & VGG16
- **Dataset** — statistik jumlah gambar per kelas
- **Prediction** — upload gambar (drag & drop), preview, predict, reset
- **Result** — nama buah, status Fresh/Rotten, confidence score, probabilitas semua kelas
- **Model Info** — arsitektur, jumlah parameter, akurasi validasi

Jika confidence prediksi < 70%, akan muncul peringatan agar pengguna
mengunggah gambar yang lebih jelas.

## Deployment ke Vercel

Model TensorFlow/VGG16 berukuran cukup besar (>100MB), sedangkan Vercel
Serverless Function punya batas ukuran ~250MB (unzipped) dan tidak
menyediakan GPU/long-running process. Project ini **sudah dilengkapi**
`vercel.json`, `requirements.txt`, dan `runtime.txt` sehingga bisa dicoba
di-deploy, tapi untuk penggunaan produksi disarankan platform yang mendukung
container penuh seperti **Railway**, **Render**, atau **Google Cloud Run**
agar model TensorFlow dapat berjalan tanpa batasan ukuran/cold-start.

```bash
vercel --prod
```

## Konfigurasi Model

- Base model: VGG16 (`include_top=False`, `weights='imagenet'`), seluruh layer konvolusi di-*freeze*
- Classifier: `GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.5) → Dense(6, softmax)`
- Optimizer: Adam, learning rate `0.0001`
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Epoch: 25, Batch size: 16
