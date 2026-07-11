import sys
import types
import os
import shutil

# Mock tensorflow_hub before importing tensorflowjs so it doesn't crash on TF 2.16+
dummy = types.ModuleType('tensorflow_hub')
dummy.load = lambda *a, **k: None
dummy.KerasLayer = object
dummy.resolve = lambda *a, **k: None
sys.modules['tensorflow_hub'] = dummy
sys.modules['tensorflow_hub.estimator'] = dummy

import tensorflow as tf
import tensorflowjs as tfjs

print("Loading Keras 3 model...")
model = tf.keras.models.load_model('dust_detector_model.keras')

saved_model_dir = 'temp_saved_model'
if os.path.exists(saved_model_dir):
    shutil.rmtree(saved_model_dir)

print("Exporting to TensorFlow SavedModel...")
model.export(saved_model_dir)

out_dir = os.path.join('static', 'tfjs_model')
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

print("Converting SavedModel to TensorFlow.js web graph model...")
tfjs.converters.convert_tf_saved_model(
    saved_model_dir,
    out_dir
)

print("Cleaning up temp saved model...")
if os.path.exists(saved_model_dir):
    shutil.rmtree(saved_model_dir)

print("SUCCESS! TFJS model generated in:", out_dir)
