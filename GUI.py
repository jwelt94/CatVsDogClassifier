import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Progressbar, Style
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import matplotlib as mpl
from keras.models import load_model

# =========================
# CONFIG
# =========================
MODEL_NAME = "CatsOrDogsModel.keras"
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "last_conv"

CLASSES = {
    0: "Image is of a Cat 🐱",
    1: "Image is of a Dog 🐶"
}

# UI COLORS
BG_COLOR = "#0f172a"
CARD_COLOR = "#111827"
ACCENT = "#38bdf8"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_NAME)

# =========================
# IMAGE HELPERS
# =========================
def get_img_array(img_path, size):
    img = tf.keras.utils.load_img(img_path, target_size=size)
    array = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(array, axis=0)

# =========================
# GRAD-CAM
# =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = tf.pow(heatmap, 0.5)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def save_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.6):
    img = tf.keras.utils.load_img(img_path)
    img = tf.keras.utils.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    superimposed = jet_heatmap * alpha + img
    superimposed = tf.keras.utils.array_to_img(superimposed)
    superimposed.save(cam_path)

# =========================
# PREDICTION PIPELINE
# =========================
def predict_and_generate_cam(img_path):
    img_array = get_img_array(img_path, IMG_SIZE)

    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    confidence = float(preds[0][pred_index])

    heatmap = make_gradcam_heatmap(
        img_array, model, LAST_CONV_LAYER, pred_index
    )
    save_gradcam(img_path, heatmap)

    return CLASSES[pred_index], confidence, "cam.jpg"

# =========================
# TKINTER UI
# =========================
top = tk.Tk()
top.title("Cat vs Dog Classifier")
top.geometry("1000x800")
top.configure(bg=BG_COLOR)

# HEADER
heading = tk.Label(
    top,
    text="🐱 Cat vs Dog Classifier 🐶",
    font=("Segoe UI", 28, "bold"),
    bg=BG_COLOR,
    fg=ACCENT
)
heading.pack(pady=(20, 10))

# IMAGE CARD
image_card = tk.Frame(
    top,
    bg=CARD_COLOR,
    highlightthickness=1,
    highlightbackground="#1f2933"
)
image_card.pack(pady=20)

image_label = tk.Label(image_card, bg=CARD_COLOR)
image_label.pack(padx=20, pady=20)

# PREDICTION LABEL
prediction_label = tk.Label(
    top,
    text="Upload an image to begin",
    font=("Segoe UI", 20, "bold"),
    bg=BG_COLOR,
    fg=TEXT
)
prediction_label.pack(pady=10)

# CONFIDENCE BAR
style = Style()
style.theme_use("default")
style.configure(
    "Confidence.Horizontal.TProgressbar",
    troughcolor=CARD_COLOR,
    background=ACCENT,
    thickness=20
)

confidence_bar = Progressbar(
    top,
    style="Confidence.Horizontal.TProgressbar",
    orient="horizontal",
    length=400,
    mode="determinate"
)
confidence_bar.pack(pady=10)

# STATUS
status_label = tk.Label(
    top,
    text="Ready",
    font=("Segoe UI", 10),
    bg=BG_COLOR,
    fg=MUTED
)
status_label.pack(pady=5)

# =========================
# BUTTON ACTION
# =========================
def upload_image():
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )
    if not path:
        return

    status_label.config(text="Analyzing image…")
    top.update_idletasks()

    pred_text, confidence, cam_path = predict_and_generate_cam(path)

    prediction_label.config(
        text=pred_text,
        fg="#22c55e" if "Cat" in pred_text else "#f97316"
    )

    confidence_bar["value"] = confidence * 100

    cam_img = Image.open(cam_path)
    cam_img.thumbnail((600, 600))
    tk_img = ImageTk.PhotoImage(cam_img)

    image_label.configure(image=tk_img)
    image_label.image = tk_img

    status_label.config(
        text=f"Confidence: {confidence:.2%}"
    )

# UPLOAD BUTTON
upload_button = tk.Button(
    top,
    text="📸 Upload Image",
    command=upload_image,
    font=("Segoe UI", 14, "bold"),
    bg=ACCENT,
    fg="black",
    activebackground="#0ea5e9",
    relief=tk.FLAT,
    padx=20,
    pady=10,
    cursor="hand2"
)
upload_button.pack(pady=30)

top.mainloop()
