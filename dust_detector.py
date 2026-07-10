"""
dust_detector.py — Urban AI Solar Panel Dust Detection
=======================================================
Trains an InceptionV3-based binary classifier to detect
dust on solar panel images.

Steps:
  1. Reads real Clean images from archive (1)/Detect_solar_dust/Clean/
  2. Synthesises matching Dusty images via augmentation
  3. Trains InceptionV3 with transfer learning (top layers first, then fine-tune)
  4. Saves model as dust_detector_model.keras
  5. Copies sample images for the Flask demo
"""

import os
import shutil
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import cv2

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent
DATASET_DIR  = ROOT / "archive (1)" / "Detect_solar_dust"
CLEAN_DIR    = DATASET_DIR / "Clean"
DUSTY_DIR    = DATASET_DIR / "Dusty"
MODEL_PATH   = ROOT / "dust_detector_model.keras"
SAMPLE_CLEAN = ROOT / "sample_clean.jpg"
SAMPLE_DUSTY = ROOT / "sample_dusty.jpg"

IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 16
EPOCHS_TOP   = 12   # Phase 1: train custom head only
EPOCHS_FINE  = 8    # Phase 2: fine-tune last 30 InceptionV3 layers
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)

# ════════════════════════════════════════════════════════════════════════════
# STEP 1: Synthesise Dusty Images
# ════════════════════════════════════════════════════════════════════════════

def synthesise_dusty(img_pil: Image.Image) -> Image.Image:
    """
    Transform a clean solar panel image into a realistic synthetic dusty one.
    Applies a randomised combination of:
      - Gaussian blur          (dust haze / diffusion)
      - Yellow-brown tint      (dust colour)
      - Brightness reduction   (dust blocks light)
      - Contrast reduction     (dust diffuses contrast)
      - Gaussian noise         (dust grain texture)
      - Optional vignette      (edge dust accumulation)
    """
    img = img_pil.convert("RGB")

    # 1. Gaussian blur (dust diffusion effect)
    blur_radius = random.uniform(0.8, 2.5)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 2. Reduce brightness (dust blocks sunlight)
    brightness_factor = random.uniform(0.60, 0.85)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # 3. Reduce contrast (dust flattens image)
    contrast_factor = random.uniform(0.65, 0.90)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # 4. Reduce colour saturation (dust desaturates)
    saturation_factor = random.uniform(0.50, 0.80)
    img = ImageEnhance.Color(img).enhance(saturation_factor)

    # 5. Yellow-brown tint overlay (typical dust colour)
    tint_strength = random.uniform(0.18, 0.38)
    tint_color    = (
        int(random.uniform(200, 230)),   # R — warm
        int(random.uniform(170, 200)),   # G — earthy
        int(random.uniform(80,  130))    # B — low blue
    )
    tint_layer = Image.new("RGB", img.size, tint_color)
    img = Image.blend(img, tint_layer, alpha=tint_strength)

    # 6. Gaussian noise (dust grain texture)
    arr = np.array(img, dtype=np.float32)
    noise_std = random.uniform(8, 22)
    noise = np.random.normal(0, noise_std, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # 7. Optional vignette (dust tends to accumulate at edges)
    if random.random() > 0.4:
        w, h = img.size
        vignette = Image.new("L", (w, h), 0)
        pixels = vignette.load()
        cx, cy = w // 2, h // 2
        max_dist = (cx ** 2 + cy ** 2) ** 0.5
        strength = random.uniform(0.12, 0.30)
        for y in range(h):
            for x in range(w):
                dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                fade = int(255 * strength * (dist / max_dist) ** 1.5)
                pixels[x, y] = min(fade, 255)
        # Blend dark vignette onto the image
        rgb_arr = np.array(img, dtype=np.float32)
        vignette_arr = np.array(vignette, dtype=np.float32) / 255.0
        for c in range(3):
            rgb_arr[:, :, c] = np.clip(
                rgb_arr[:, :, c] * (1 - vignette_arr), 0, 255
            )
        img = Image.fromarray(rgb_arr.astype(np.uint8))

    return img


def generate_dusty_dataset():
    """Create Dusty/ by synthesising from every Clean/ image."""
    DUSTY_DIR.mkdir(parents=True, exist_ok=True)
    clean_images = list(CLEAN_DIR.glob("*.jpg")) + list(CLEAN_DIR.glob("*.png"))

    if not clean_images:
        raise FileNotFoundError(f"No images found in {CLEAN_DIR}")

    print(f"\n[INFO] Found {len(clean_images)} clean images.")
    print(f"[INFO] Generating synthetic dusty images -> {DUSTY_DIR}")

    for idx, clean_path in enumerate(clean_images):
        # Name e.g. Imgclean_146_0.jpg  →  Imgdust_146_0.jpg
        dusty_name = clean_path.name.replace("Imgclean", "Imgdust", 1)
        dusty_path = DUSTY_DIR / dusty_name

        if dusty_path.exists():
            continue  # skip if already generated

        try:
            clean_img = Image.open(clean_path).convert("RGB")
            dusty_img = synthesise_dusty(clean_img)
            dusty_img.save(dusty_path, quality=92)
            if (idx + 1) % 30 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(clean_images)}] {dusty_name}")
        except Exception as exc:
            print(f"  [SKIP] {clean_path.name}: {exc}")

    generated = len(list(DUSTY_DIR.glob("*.jpg")))
    print(f"[DONE] Dusty dataset ready - {generated} images in {DUSTY_DIR}\n")
    return clean_images


# ════════════════════════════════════════════════════════════════════════════
# STEP 2: Build TensorFlow Dataset
# ════════════════════════════════════════════════════════════════════════════

def build_datasets():
    import tensorflow as tf

    print("[INFO] Building TensorFlow image datasets ...")

    datagen_args = dict(
        rescale            = None,          # handled by preprocess_input
        validation_split   = 0.20,
        horizontal_flip    = True,
        vertical_flip      = True,
        rotation_range     = 20,
        zoom_range         = 0.15,
        width_shift_range  = 0.10,
        height_shift_range = 0.10,
        brightness_range   = (0.80, 1.20),
        fill_mode          = "nearest",
        preprocessing_function = (
            tf.keras.applications.inception_v3.preprocess_input
        ),
    )

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_gen = ImageDataGenerator(**datagen_args)
    val_gen   = ImageDataGenerator(
        rescale          = None,
        validation_split = 0.20,
        preprocessing_function = (
            tf.keras.applications.inception_v3.preprocess_input
        ),
    )

    train_ds = train_gen.flow_from_directory(
        str(DATASET_DIR),
        target_size  = IMAGE_SIZE,
        batch_size   = BATCH_SIZE,
        class_mode   = "categorical",
        subset       = "training",
        seed         = SEED,
        shuffle      = True,
    )
    val_ds = val_gen.flow_from_directory(
        str(DATASET_DIR),
        target_size  = IMAGE_SIZE,
        batch_size   = BATCH_SIZE,
        class_mode   = "categorical",
        subset       = "validation",
        seed         = SEED,
        shuffle      = False,
    )

    print(f"  Classes      : {train_ds.class_indices}")
    print(f"  Train samples: {train_ds.samples}")
    print(f"  Val   samples: {val_ds.samples}\n")
    return train_ds, val_ds


# ════════════════════════════════════════════════════════════════════════════
# STEP 3: Build Model
# ════════════════════════════════════════════════════════════════════════════

def build_model():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    print("[INFO] Building InceptionV3 transfer learning model ...")

    base = keras.applications.InceptionV3(
        weights     = "imagenet",
        include_top = False,
        input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    base.trainable = False   # freeze all base layers initially

    inputs  = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.50)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    print(f"  Total params  : {model.count_params():,}")
    print(f"  Trainable     : {sum(p.numpy().size for p in model.trainable_weights):,}\n")
    return model, base


# ════════════════════════════════════════════════════════════════════════════
# STEP 4: Train
# ════════════════════════════════════════════════════════════════════════════

def train(model, base, train_ds, val_ds):
    import tensorflow as tf
    from tensorflow import keras

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            monitor        = "val_accuracy",
            save_best_only = True,
            verbose        = 1,
        ),
        keras.callbacks.EarlyStopping(
            monitor   = "val_loss",
            patience  = 4,
            restore_best_weights = True,
            verbose   = 1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.4,
            patience = 2,
            min_lr   = 1e-7,
            verbose  = 1,
        ),
    ]

    # ── Phase 1: train custom head only ─────────────────────────────────────
    print("=" * 60)
    print("PHASE 1 — Training custom head  (base frozen)")
    print("=" * 60)
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-3),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data  = val_ds,
        epochs           = EPOCHS_TOP,
        callbacks        = callbacks,
        verbose          = 1,
    )

    # ── Phase 2: fine-tune last 30 layers of InceptionV3 ────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 — Fine-tuning last 30 InceptionV3 layers")
    print("=" * 60)
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-4),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data  = val_ds,
        epochs           = EPOCHS_FINE,
        callbacks        = callbacks,
        verbose          = 1,
    )

    print(f"\n[DONE] Model saved -> {MODEL_PATH}")
    return model


# ════════════════════════════════════════════════════════════════════════════
# STEP 5: Evaluate & extract sample images
# ════════════════════════════════════════════════════════════════════════════

def evaluate_and_save_samples(model, val_ds):
    import tensorflow as tf
    from tensorflow import keras

    print("\n[INFO] Evaluating on validation set ...")
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"  Val loss     : {loss:.4f}")
    print(f"  Val accuracy : {acc:.4f}  ({acc*100:.1f}%)")

    # Confusion matrix
    val_ds.reset()
    preds = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_ds.classes
    class_names = {v: k for k, v in val_ds.class_indices.items()}

    from collections import Counter
    print("\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':10s}  {'Clean':>8s}  {'Dusty':>8s}")
    for true_cls in [0, 1]:
        tp_clean = sum(1 for t, p in zip(y_true, y_pred) if t == true_cls and p == 0)
        tp_dusty  = sum(1 for t, p in zip(y_true, y_pred) if t == true_cls and p == 1)
        print(f"  {class_names[true_cls]:10s}  {tp_clean:>8d}  {tp_dusty:>8d}")

    # Copy sample images
    clean_files = sorted(CLEAN_DIR.glob("*.jpg"))
    dusty_files = sorted((DATASET_DIR / "Dusty").glob("*.jpg"))

    if clean_files:
        shutil.copy(clean_files[0], SAMPLE_CLEAN)
        print(f"  sample_clean.jpg -> copied from {clean_files[0].name}")
    if dusty_files:
        shutil.copy(dusty_files[0], SAMPLE_DUSTY)
        print(f"  sample_dusty.jpg -> copied from {dusty_files[0].name}")

    print("\n[DONE] All done! Run:  python app.py")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Urban AI - Solar Panel Dust Detector Training")
    print("=" * 60)

    # 1. Generate synthetic dusty dataset
    clean_images = generate_dusty_dataset()

    # 2. Build TF datasets
    train_ds, val_ds = build_datasets()

    # 3. Build model
    model, base = build_model()

    # 4. Train
    model = train(model, base, train_ds, val_ds)

    # 5. Evaluate & save samples
    evaluate_and_save_samples(model, val_ds)
