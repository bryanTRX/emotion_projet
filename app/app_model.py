import cv2
import mediapipe as mp
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from utils import load_model, preprocess_image, get_emotion_label

def run_realtime_emotion_detection(
    model_path: str = "../models/best_model.pth",
    camera_index: int = 0,
    min_confidence: float = 0.5
):
    print("Initialisation du modele et de la camera...")
    model, device = load_model(model_path)
    model.eval()

    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=min_confidence)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la camera.")

    print("Caméra demarree — appuyez sur 'q' pour quitter.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape

                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue
                tensor = preprocess_image(face_crop).to(device)

                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)

                label = get_emotion_label(pred.item())
                confidence = conf.item() * 100

                color = (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ({confidence:.1f}%)",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        cv2.imshow("Detection d'emotions en temps reel", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Session terminée proprement.")

if __name__ == "__main__":
    run_realtime_emotion_detection()
