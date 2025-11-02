import streamlit as st
import cv2
import mediapipe as mp
import torch
import numpy as np
import sys
import os
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from utils import load_model, preprocess_image, get_emotion_label

st.set_page_config(page_title="Détection d'émotions", layout="wide")
st.title("🧠 Détection d'Émotions en Temps Réel")

# --- Sidebar paramètres ---
st.sidebar.header("⚙️ Paramètres")
camera_index = st.sidebar.selectbox("Caméra", [0, 1, 2], index=0)
confidence_threshold = st.sidebar.slider("Seuil de confiance (%)", 0, 100, 50) / 100.0
model_path = st.sidebar.text_input("Chemin du modèle", "models/best_model.pth")

# --- Charger le modèle ---
@st.cache_resource
def load_emotion_model(path):
    model, device = load_model(path)
    model.eval()
    return model, device

model, device = load_emotion_model(model_path)
st.sidebar.success(f"✅ Modèle chargé depuis {model_path}")

# --- Initialisation MediaPipe ---
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=confidence_threshold)

# --- Section vidéo ---
stframe = st.empty()
emotion_text = st.empty()
emotion_confidence = st.empty()
emotion_image = st.empty()

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    st.error(f"❌ Impossible d'ouvrir la caméra {camera_index}")

quit_btn = st.sidebar.button("Quitter")

try:
    while True:
        if quit_btn:
            st.warning("🛑 Détection arrêtée par l'utilisateur")
            break

        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Impossible de lire la frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, int((bbox.xmin + bbox.width) * w))
                y2 = min(h, int((bbox.ymin + bbox.height) * h))

                face_crop = frame[y1:y2, x1:x2].copy()
                if face_crop.size == 0:
                    continue

                tensor = preprocess_image(face_crop).to(device)
                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)

                label = get_emotion_label(int(pred.item()))
                confidence = float(conf.item())

                color = (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ({confidence*100:.1f}%)",
                    (x1, max(15, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2
                )

                emotion_text.markdown(f"### 🎭 Émotion détectée : {label}")
                emotion_confidence.markdown(f"Confiance : {confidence*100:.1f}%")
                safe_name = "".join(c for c in label.lower() if c.isalnum())
                try:
                    img_path = f"app/assets/emotions/{safe_name}.png"
                    pil_img = Image.open(img_path).resize((150,150))
                    emotion_image.image(pil_img)
                except:
                    emotion_image.empty()

        stframe.image(frame, channels="BGR")

finally:
    cap.release()
