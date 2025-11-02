import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import mediapipe as mp
import torch
import numpy as np
import sys
import os
from PIL import Image
import cv2

st.set_page_config(page_title="Détection d'émotions", layout="wide")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from utils import load_model, preprocess_image, get_emotion_label

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.sidebar.header("⚙️ Paramètres")
confidence_threshold = st.sidebar.slider("Seuil de confiance (%)", 0, 100, 50) / 100.0
model_path = st.sidebar.text_input("Chemin du modèle", "models/best_model.pth")

@st.cache_resource
def load_emotion_model(path):
    model, device = load_model(path)
    model.eval()
    return model, device

model, device = load_emotion_model(model_path)
st.sidebar.success(f"✅ Modèle chargé depuis {model_path}")

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=confidence_threshold
        )
        self.current_emotion = "Aucune"
        self.current_confidence = 0.0
        self.current_image = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = img.shape
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, int((bbox.xmin + bbox.width) * w))
                y2 = min(h, int((bbox.ymin + bbox.height) * h))

                face_crop = img[y1:y2, x1:x2].copy()
                if face_crop.size > 0:
                    tensor = preprocess_image(face_crop).to(device)
                    with torch.no_grad():
                        outputs = model(tensor)
                        probs = torch.softmax(outputs, dim=1)
                        conf, pred = torch.max(probs, 1)

                    label = get_emotion_label(int(pred.item()))
                    confidence = float(conf.item())

                    self.current_emotion = label
                    self.current_confidence = confidence

                    # Image associée
                    safe_name = "".join(c for c in label.lower() if c.isalnum())
                    try:
                        img_path = f"app/assets/emotions/{safe_name}.png"
                        self.current_image = Image.open(img_path).resize((200, 200))
                    except:
                        self.current_image = None

                    color = (0, 255, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    text = f"{label} ({confidence*100:.1f}%)"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                    cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🧠 Détection d'Émotions en Temps Réel")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Flux vidéo")
    ctx = webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("📊 Résultats")
    emotion_placeholder = st.empty()
    confidence_placeholder = st.empty()
    image_placeholder = st.empty()

# --- Boucle de mise à jour ---
import time
while ctx.state.playing:
    if ctx.video_processor:
        emotion = ctx.video_processor.current_emotion
        confidence = ctx.video_processor.current_confidence
        pil_img = ctx.video_processor.current_image

        emotion_placeholder.markdown(f"### 🎭 **{emotion}**")
        confidence_placeholder.progress(confidence)
        confidence_placeholder.markdown(f"Confiance : **{confidence*100:.1f}%**")

        if pil_img:
            image_placeholder.image(pil_img)
        else:
            image_placeholder.empty()
    time.sleep(0.1)
