# 🚦 RoadGuard v2

> AI-powered Traffic Sign Recognition using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Flask](https://img.shields.io/badge/Flask-API-green) ![Accuracy](https://img.shields.io/badge/Accuracy-97.7%25-brightgreen)

---

## 📋 What It Does

RoadGuard identifies **43 types of traffic signs** from an uploaded image using a trained Convolutional Neural Network. It shows:

- **Top-3 predictions** with confidence scores
- **Sign description** (what the sign means)
- **Grad-CAM heatmap** (where the AI looked)
- **Live webcam detection**
- **Batch processing** (many images at once)

---

## 📁 Project Structure

```
roadgaurd/
├── data/                   ← GTSRB dataset (Train/, Test/, Test.csv)
├── saved_models/
│   ├── custom_cnn.h5       ← Trained CNN model
│   └── custom_cnn.tflite   ← TFLite model (after export)
├── backend/
│   ├── app.py              ← Flask API server
│   └── model_loader.py     ← Model loading + prediction + Grad-CAM
├── frontend/
│   ├── index.html          ← Main web app
│   ├── gallery.html        ← All 43 sign classes reference
│   ├── style.css           ← Styling (sky blue + orange theme)
│   └── script.js           ← Frontend logic
├── preprocess.py           ← Data loading & preprocessing
├── models.py               ← CNN & MobileNetV2 architecture
├── train.py                ← Training script
├── evaluate.py             ← Evaluation & plots
├── main.py                 ← Run full pipeline
├── export_tflite.py        ← Convert model to TFLite
└── requirements.txt        ← Python dependencies
```

---

## ⚙️ Installation

### 1. Clone / Download the project

```bash
cd C:\Users\SAMPATH\roadgaurd
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download **GTSRB** from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and extract to `data/` folder.

```
data/
├── Train/      (43 class folders)
├── Test/       (test images)
└── Test.csv
```

---

## 🚀 How to Run

### Step 1 — Train the model (first time only)

```bash
python main.py
```

This will train both models and save `saved_models/custom_cnn.h5`.

### Step 2 — Start Flask backend

```bash
python backend/app.py
```

You should see:
```
model loaded successfully
* Running on http://127.0.0.1:5000
```

### Step 3 — Open the web app

Double-click `frontend/index.html` in File Explorer.

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/predict` | POST | Single image prediction (top-3) |
| `/predict_batch` | POST | Multiple images prediction |
| `/gradcam` | POST | Grad-CAM heatmap |

### Example — `/predict`

```bash
curl -X POST http://localhost:5000/predict -F "file=@sign.png"
```

Response:
```json
{
  "top": [
    {"class_id": 14, "label": "Stop", "description": "...", "confidence": 99.8},
    {"class_id": 17, "label": "No entry", "description": "...", "confidence": 0.1},
    {"class_id": 13, "label": "Yield", "description": "...", "confidence": 0.05}
  ]
}
```

---

## 📊 Model Performance

| Model | Test Accuracy | Parameters |
|---|---|---|
| Custom CNN | **97.7%** | ~1.1M |
| MobileNetV2 | Transfer learning | ~2.3M |

**Dataset:** GTSRB (German Traffic Sign Recognition Benchmark)
- Training: 39,209 images
- Test: 12,630 images
- Classes: 43

---

## 📱 Export for Mobile (TFLite)

```bash
python export_tflite.py
```

Creates `saved_models/custom_cnn.tflite` — optimized for Android/iOS/Raspberry Pi.

---

## 🔧 Requirements

```
tensorflow>=2.11
flask
flask-cors
opencv-python
numpy
pandas
scikit-learn
matplotlib
seaborn
pillow
```

---

## 📸 Features

| Feature | Description |
|---|---|
| 📷 Upload | Drag & drop or browse image |
| 🎥 Webcam | Live camera capture & detect |
| 📁 Batch | Process many images at once |
| 🔥 Grad-CAM | Heatmap showing AI focus area |
| 📋 History | Last 20 detections logged |
| 🌙 Dark mode | Toggle light/dark theme |
| ⚠️ Warnings | Alert for low confidence |
| 🔔 Sound | Beep for dangerous signs |
| 📄 CSV export | Download batch results |
| 🖼 Gallery | Reference for all 43 signs |

---

## 📄 License

MIT License — Free to use for education and personal projects.
