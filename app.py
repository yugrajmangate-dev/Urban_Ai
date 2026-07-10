import os
import base64
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import tensorflow as tf
from tensorflow import keras

# Define paths
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "dust_detector_model.keras"
SAMPLE_CLEAN_PATH = ROOT / "sample_clean.jpg"
SAMPLE_DUSTY_PATH = ROOT / "sample_dusty.jpg"

# Configure TensorFlow settings for stable inference
tf.config.set_visible_devices([], 'GPU') # CPU inference is fine and stable for single images

print("Loading TensorFlow InceptionV3 model from:", MODEL_PATH)
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure it is trained first.")

# Load the trained model
model = keras.models.load_model(MODEL_PATH)
class_names = ["Clean", "Dusty"]
IMAGE_SIZE = (224, 224)
print("Model loaded successfully. Ready for inference.")

# Initialize Flask
app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sample/clean')
def get_sample_clean():
    if not SAMPLE_CLEAN_PATH.exists():
        return "Clean sample image not found", 404
    return send_file(SAMPLE_CLEAN_PATH, mimetype='image/jpeg')

@app.route('/sample/dusty')
def get_sample_dusty():
    if not SAMPLE_DUSTY_PATH.exists():
        return "Dusty sample image not found", 404
    return send_file(SAMPLE_DUSTY_PATH, mimetype='image/jpeg')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        # Handle base64 image input (webcam or uploaded)
        if 'image' in data:
            image_data = data['image']
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Handle demo image requests
        elif 'source' in data:
            source = data['source']
            if source == 'clean':
                img = cv2.imread(str(SAMPLE_CLEAN_PATH))
            elif source == 'dusty':
                img = cv2.imread(str(SAMPLE_DUSTY_PATH))
            else:
                return jsonify({"error": "Unknown sample source"}), 400
        else:
            return jsonify({"error": "No image data or source specified"}), 400

        if img is None:
            return jsonify({"error": "Failed to load/decode image"}), 400

        # Process the image for InceptionV3
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, IMAGE_SIZE)
        prepared = keras.applications.inception_v3.preprocess_input(resized.astype("float32"))
        
        # Run inference
        pred = model.predict(np.expand_dims(prepared, axis=0), verbose=0)[0]
        idx = int(np.argmax(pred))
        confidence = float(pred[idx]) * 100
        label = class_names[idx]

        return jsonify({
            "status": label,
            "confidence": confidence,
            "success": True
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500

if __name__ == '__main__':
    # Run server locally on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
