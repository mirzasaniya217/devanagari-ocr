# Devanagari OCR API

A production-ready Optical Character Recognition (OCR) system for **Devanagari script**, built using deep learning and exposed as a secure REST API.

This project performs **line detection, character segmentation, and classification** with honest confidence scoring, suitable for real-world and academic use.

---

## Live Deployment

- **API Base URL:**  
  https://devanagari-ocr.onrender.com

- **Interactive API Docs (Swagger):**  
  https://devanagari-ocr.onrender.com/docs

Upload an image and receive recognized Devanagari text with per-character confidence.

---

##  Features

- Line detection and segmentation
- Character-level OCR for Devanagari script
- Supports **57 Devanagari classes** (consonants, vowels, numerals)
- EfficientNet-based deep learning model
- Honest confidence estimation (non-fabricated)
- RESTful API using FastAPI
- Secure file handling and cleanup
- Cloud-deployed on Render

---

## Model & Approach

- **Architecture:** EfficientNet (transfer learning)
- **Input:** Image (PNG / JPEG)
- **Output:**  
  - Recognized text (multi-line supported)  
  - Character-wise predictions with confidence
- **Confidence Strategy:**  
  Relative probability normalization between top predictions (no artificial inflation)

---

##  Tech Stack

- **Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Computer Vision:** OpenCV  
- **Backend Framework:** FastAPI  
- **Model Architecture:** EfficientNet  
- **Deployment:** Render Cloud  

---

##  API Usage
### Request
- `multipart/form-data`
- Upload an image file (`.png`, `.jpg`)

### Response (Example)
```json
{
  "recognized_text": "२२२२२२२२",
  "characters": [
    {
      "position": 1,
      "predicted": "२",
      "candidates": [
        { "char": "२", "confidence": 50.47 },
        { "char": "९", "confidence": 49.53 }
      ]
    }
  ]
}

### Endpoint
