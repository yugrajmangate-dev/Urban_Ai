# 🌟 Urban AI — Solar Panel Dust Detection System

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-InceptionV3-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![TensorFlow.js](https://img.shields.io/badge/TensorFlow.js-Browser_AI-FFA800?style=for-the-badge&logo=javascript&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-4ade80?style=for-the-badge)

An enterprise-grade, real-time AI computer vision system designed to detect dust accumulation on solar panels using transfer learning on Google's **InceptionV3** architecture. Features a stunning modern glassmorphism web dashboard with support for **Webcam live capture**, **Drag & Drop image uploads**, and **Client-Side Browser AI inference (`TensorFlow.js`)**.

---

## ❓ Frequently Asked Question: *Do my friends need to install 1.2 GB of TensorFlow or download heavy model files to run this locally?*

### 🛑 **NO! Absolutely NOT!**
We have converted our trained deep neural network (`InceptionV3`) into a modular **TensorFlow.js (`tfjs`)** browser format located in `static/tfjs_model/`. 
* **Zero 1.2 GB Downloads Required**: The entire AI engine (`86 MB` sharded across 23 small git files) is already included right inside this repository when you clone it!
* **Zero Python Dependencies Needed (for Option 1)**: The model runs directly inside your web browser using your computer's GPU/WebGL acceleration!

---

## 🚀 How to Run on Localhost (Choose Option 1 or Option 2)

### Option 1: Ultra-Fast Browser AI Mode (Recommended — 10 Seconds — No Python Needed!)
You do **not** need to install Python, `pip`, or `requirements.txt` to run the app locally.

1. **Clone or Download the Repository:**
   ```bash
   git clone https://github.com/yugrajmangate-dev/Urban_Ai.git
   cd Urban_Ai
   ```
2. **Launch a Quick Local Static Server:**
   * **If using Python:**
     ```bash
     python -m http.server 8000
     ```
   * **If using Node.js (`npx`):**
     ```bash
     npx serve .
     ```
   * **If using VS Code:** Just right-click `index.html` and click **"Open with Live Server"**.
3. **Open in Browser:** Visit `http://localhost:8000` (or `http://localhost:3000`). The status badge will say **`⬤ Model Ready (Browser AI)`** and AI detection will run instantly on your local device!

---

### Option 2: Python Flask Backend Mode (For Developers)
If you specifically want to run the Flask Python server and API endpoints (`/predict`):

1. **Clone the Repository & Create Virtual Environment:**
   ```bash
   git clone https://github.com/yugrajmangate-dev/Urban_Ai.git
   cd Urban_Ai
   python -m venv venv
   ```
2. **Activate the Virtual Environment:**
   * **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
   * **Windows (CMD):** `.\venv\Scripts\activate.bat`
   * **Mac / Linux:** `source venv/bin/activate`
3. **Install Lightweight Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: If you do not have `dust_detector_model.keras` locally due to `.gitignore` exclusions, the server starts cleanly and automatically delegates inference to the included client-side `static/tfjs_model` engine!)*
4. **Start the Flask Server:**
   ```bash
   python app.py
   ```
5. **Open in Browser:** Visit `http://localhost:5000`.

---

## 🔬 Model & Performance Highlights

* **Base Architecture**: `InceptionV3` (pre-trained on ImageNet, fine-tuned via transfer learning).
* **Custom Classifier Head**: `GlobalAveragePooling2D` → `Dense(128, ReLU)` → `Dropout(0.3)` → `Dense(2, Softmax)`.
* **Validation Accuracy**: **92.65%** across augmented synthetic clean and dusty solar panel test sets.
* **Inference Speed**: `< 100 milliseconds` per image via WebGL Client-Side Acceleration or local Python CPU threads.

---

## 🎨 UI / UX Features

* **Glassmorphism Aesthetic**: Curated dark-mode HSL color tokens, neon glow highlights, and smooth CSS cubic-bezier transitions.
* **Interactive Particle Background**: Dynamic floating dust/light particles rendered on HTML5 `<canvas>`.
* **Real-Time Webcam Analysis**: Instantly capture frames from any laptop or USB webcam for on-site inspection.
* **Live Confidence Ring**: SVG circular progress bar with animated stroke transitions detailing AI certainty.
* **Prediction History Log**: Tracks up to 10 previous inspection runs with color-coded classification pills.

---

## 📁 Repository Structure

```text
Urban_Ai/
├── app.py                      # Flask API Server & Backend Controller
├── dust_detector.py            # Deep Learning Data Pipeline & Training Script
├── index.html                  # Universal Root Entry Point for Static/Vercel Hosting
├── sample_clean.jpg            # Demo Sample Image (Clean Panel)
├── sample_dusty.jpg            # Demo Sample Image (Dusty Panel)
├── requirements.txt            # Python Backend Dependencies
├── static/
│   ├── style.css               # Modern Glassmorphism Design System
│   ├── app.js                  # Frontend Controller + Client-Side TFJS AI Engine
│   └── tfjs_model/             # Browser-Optimized Sharded AI Weights (NO INSTALL NEEDED)
│       ├── model.json          # Network Architecture & Layer Topology
│       └── group1-shard*.bin   # 23 Binary Shards (< 4 MB each for Git compatibility)
└── templates/
    └── index.html              # Flask Template View
```

---

## 🤝 Contributing & License
Created and maintained by [Yugraj Mangate](https://github.com/yugrajmangate-dev) and collaborators.  
Licensed under the **MIT License**.
