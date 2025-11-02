import sys
import os
import cv2
import mediapipe as mp
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider,
                             QComboBox, QGroupBox, QTextEdit)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from utils import load_model, preprocess_image, get_emotion_label


class VideoThread(QThread):
    """Thread pour traiter la vidéo sans bloquer l'interface"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    emotion_signal = pyqtSignal(str, float)
    log_signal = pyqtSignal(str)  # Nouveau signal pour les logs
    error_signal = pyqtSignal(str)  # Signal pour les erreurs

    def __init__(self, model_path, camera_index=0, min_confidence=0.5):
        super().__init__()
        self.model_path = model_path
        self.camera_index = camera_index
        self.min_confidence = float(min_confidence)
        self.running = False
        self.paused = False
        self.cap = None
        self.detector = None
        self.model = None
        self.device = None

    def log(self, message):
        """Envoie un message de log à l'interface"""
        print(message, flush=True)
        self.log_signal.emit(message)

    def run(self):
        """Boucle principale du thread vidéo"""
        try:
            self.log("🔄 Chargement du modèle...")
            try:
                self.model, self.device = load_model(self.model_path)
                self.model.eval()
                self.log("✅ Modèle chargé avec succès")
            except Exception as e:
                self.log(f"❌ Échec chargement modèle: {e}")
                self.error_signal.emit(f"Erreur chargement modèle:\n{e}")
                return

            self.log("🔄 Initialisation de MediaPipe...")
            mp_face = mp.solutions.face_detection
            try:
                self.detector = mp_face.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=self.min_confidence
                )
                self.log("✅ MediaPipe initialisé")
            except Exception as e:
                self.log(f"❌ Erreur initialisation MediaPipe: {e}")
                self.error_signal.emit(f"Erreur MediaPipe:\n{e}")
                return

            self.log(f"🔄 Tentative d'ouverture de la caméra {self.camera_index}...")
            self.cap = cv2.VideoCapture(self.camera_index)
            # petite pause pour laisser la caméra s'initialiser
            import time
            time.sleep(1)

            if not self.cap.isOpened():
                err = f"❌ Impossible d'ouvrir la caméra {self.camera_index}"
                self.log(err)
                self.error_signal.emit("Caméra non accessible. Vérifiez les permissions et qu'aucune autre application n'utilise la caméra.")
                return

            # test de lecture
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                self.log("❌ Caméra ouverte mais lecture des frames impossible")
                self.cap.release()
                self.error_signal.emit("Erreur de lecture vidéo")
                return

            self.log(f"✅ Caméra opérationnelle (résolution: {test_frame.shape[1]}x{test_frame.shape[0]})")
            self.running = True

            frame_count = 0
            while self.running:
                if self.paused:
                    self.msleep(100)
                    continue

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.log("❌ Erreur de lecture frame, arrêt")
                    break

                processed = self.process_frame(frame)
                # processed is BGR numpy array suitable for display
                self.change_pixmap_signal.emit(processed)

                frame_count += 1
                if frame_count == 1:
                    self.log("✅ Flux vidéo actif")

            # release resources
            if self.cap is not None:
                self.cap.release()
            self.log("✅ Caméra fermée proprement")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log(f"❌ ERREUR: {e}")
            self.log(tb)
            self.error_signal.emit(f"Erreur critique:\n{e}")
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass

    def process_frame(self, frame):
        """Traite une frame et détecte les émotions"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = None
            if self.detector is not None:
                results = self.detector.process(rgb)

            if results and getattr(results, "detections", None):
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)

                    # clamp
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    # ignore invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue

                    face_crop = frame[y1:y2, x1:x2].copy()
                    if face_crop.size == 0:
                        continue

                    # preprocess_image doit renvoyer un tensor shape (1, C, H, W)
                    try:
                        tensor = preprocess_image(face_crop).to(self.device)
                        with torch.no_grad():
                            outputs = self.model(tensor)
                            probs = torch.softmax(outputs, dim=1)
                            conf, pred = torch.max(probs, 1)
                    except Exception as e:
                        # si modèle échoue, log et continuer
                        self.log(f"⚠️ Erreur prédiction: {e}")
                        continue

                    label = get_emotion_label(int(pred.item()))
                    confidence = float(conf.item()) * 100.0

                    # Émettre le signal avec l'émotion détectée
                    self.emotion_signal.emit(label, confidence)

                    # Dessiner sur la frame (BGR)
                    color = (0, 255, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} ({confidence:.1f}%)",
                        (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

        except Exception as e:
            self.log(f"⚠️ Erreur process_frame: {e}")

        return frame

    def stop(self):
        """Arrête le thread"""
        self.log("🔄 Arrêt du thread...")
        self.running = False
        # if paused, wake it so it can exit faster
        self.paused = False
        self.wait(2000)

    def pause(self):
        """Met en pause le traitement"""
        self.paused = True
        self.log("⏸️ Détection en pause")

    def resume(self):
        """Reprend le traitement"""
        self.paused = False
        self.log("▶️ Détection reprise")

    def update_confidence(self, value):
        """Met à jour le seuil de confiance. value attendu 0-100 ou float 0.0-1.0"""
        try:
            if value > 1:
                self.min_confidence = float(value) / 100.0
            else:
                self.min_confidence = float(value)
            self.log(f"🔧 Seuil confiance mis à jour: {self.min_confidence:.2f}")
        except Exception as e:
            self.log(f"⚠️ Erreur update_confidence: {e}")


class EmotionDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Détection d'Émotions en Temps Réel")
        self.setGeometry(100, 100, 1200, 750)
        # (style sheet omitted here for brevity in this answer, keep yours as needed)
        self.setStyleSheet(""" /* conserve ton style existant */ """)

        # Variables
        self.video_thread = None
        self.is_running = False
        self.current_emotion = "Aucune"
        self.current_confidence = 0.0

        # Configuration par défaut
        self.model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../models/best_model_traced.pt"))
        self.camera_index = 0
        self.min_confidence = 0.5

        self.init_ui()

    def init_ui(self):
        """Initialise l'interface utilisateur"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Panneau gauche - Vidéo
        left_panel = QVBoxLayout()

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 3px solid #4CAF50;
                background-color: #1a1a1a;
                border-radius: 10px;
            }
        """)
        self.video_label.setText("📹 Caméra non démarrée\n\nCliquez sur 'Démarrer' pour commencer")
        left_panel.addWidget(self.video_label)

        # Boutons de contrôle
        control_layout = QHBoxLayout()

        self.start_button = QPushButton("Démarrer")
        self.start_button.clicked.connect(self.start_detection)
        control_layout.addWidget(self.start_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_detection)
        self.pause_button.setEnabled(False)
        control_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("Arrêter")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)

        left_panel.addLayout(control_layout)

        # Console de logs
        log_label = QLabel("📋 Journal d'activité:")
        left_panel.addWidget(log_label)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(120)
        self.log_console.append("Prêt à démarrer...")
        left_panel.addWidget(self.log_console)

        main_layout.addLayout(left_panel, 7)

        # Panneau droit - Paramètres et Stats
        right_panel = QVBoxLayout()

        # Groupe Émotion détectée
        emotion_group = QGroupBox("🎭 Émotion Détectée")
        emotion_layout = QVBoxLayout()

        self.emotion_display = QLabel("Aucune")
        self.emotion_display.setAlignment(Qt.AlignCenter)
        self.emotion_display.setFont(QFont("Arial", 24, QFont.Bold))
        self.emotion_display.setStyleSheet("color: #FFFFFF; padding: 20px;")
        emotion_layout.addWidget(self.emotion_display)

        self.confidence_display = QLabel("Confiance: 0.0%")
        self.confidence_display.setAlignment(Qt.AlignCenter)
        self.confidence_display.setFont(QFont("Arial", 14))
        emotion_layout.addWidget(self.confidence_display)

        # Image for emotion illustration
        self.emotion_image = QLabel()
        self.emotion_image.setAlignment(Qt.AlignCenter)
        self.emotion_image.setFixedSize(150, 150)
        self.emotion_image.setStyleSheet("background: transparent; border: none; color: white;")
        emotion_layout.addWidget(self.emotion_image, alignment=Qt.AlignCenter)

        emotion_group.setLayout(emotion_layout)
        right_panel.addWidget(emotion_group)

        # Groupe Paramètres
        settings_group = QGroupBox("⚙️ Paramètres")
        settings_layout = QVBoxLayout()

        # Sélection de caméra
        camera_label = QLabel("Caméra:")
        settings_layout.addWidget(camera_label)

        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Caméra 0", "Caméra 1", "Caméra 2"])
        self.camera_combo.currentIndexChanged.connect(self.change_camera)
        settings_layout.addWidget(self.camera_combo)

        # Slider confiance
        conf_label = QLabel("Seuil confiance (%):")
        settings_layout.addWidget(conf_label)
        conf_layout = QHBoxLayout()
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(int(self.min_confidence * 100))
        self.confidence_slider.valueChanged.connect(self.update_confidence_threshold)
        conf_layout.addWidget(self.confidence_slider)
        self.confidence_value_label = QLabel(f"{int(self.min_confidence * 100)}%")
        conf_layout.addWidget(self.confidence_value_label)
        settings_layout.addLayout(conf_layout)

        settings_group.setLayout(settings_layout)
        right_panel.addWidget(settings_group)

        # Légende des émotions
        legend_group = QGroupBox("📊 Légende des Émotions")
        legend_layout = QVBoxLayout()

        emotions = ["😠 Angry", "🤢 Disgust", "😨 Fear", "😊 Happy",
                    "😐 Neutral", "😢 Sad", "😲 Surprise"]
        for emotion in emotions:
            label = QLabel(emotion)
            label.setStyleSheet("padding: 5px;")
            legend_layout.addWidget(label)

        legend_group.setLayout(legend_layout)
        right_panel.addWidget(legend_group)

        right_panel.addStretch()
        main_layout.addLayout(right_panel, 3)

    def add_log(self, message):
        """Ajoute un message au journal"""
        self.log_console.append(message)
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def show_error(self, error_msg):
        """Affiche une erreur dans l'interface"""
        self.video_label.setText(f"❌ ERREUR\n\n{error_msg}\n\nVoir le journal ci-dessous")
        self.add_log(f"❌ ERREUR: {error_msg}")
        # arrête proprement
        self.stop_detection()

    def start_detection(self):
        """Démarre la détection d'émotions"""
        if self.is_running:
            return

        self.add_log("=" * 50)
        self.add_log("▶️ DÉMARRAGE DE LA DÉTECTION")
        self.add_log("=" * 50)

        self.video_label.setText("🔄 Initialisation en cours...")

        self.video_thread = VideoThread(
            self.model_path,
            camera_index=self.camera_index,
            min_confidence=self.min_confidence
        )
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.emotion_signal.connect(self.update_emotion)
        self.video_thread.log_signal.connect(self.add_log)
        self.video_thread.error_signal.connect(self.show_error)
        self.video_thread.start()

        self.is_running = True
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.pause_button.setText("Pause")
        self.stop_button.setEnabled(True)
        self.camera_combo.setEnabled(False)

    def pause_detection(self):
        """Met en pause/reprend la détection"""
        if not self.video_thread:
            return

        if self.video_thread.paused:
            self.video_thread.resume()
            self.pause_button.setText("Pause")
        else:
            self.video_thread.pause()
            self.pause_button.setText("Reprendre")

    def stop_detection(self):
        """Arrête la détection"""
        if self.video_thread:
            try:
                self.video_thread.stop()
            except Exception as e:
                self.add_log(f"⚠️ Erreur lors de l'arrêt du thread: {e}")
            self.video_thread = None

        self.is_running = False
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self.stop_button.setEnabled(False)
        self.camera_combo.setEnabled(True)

        self.video_label.clear()
        self.video_label.setText("📹 Caméra arrêtée\n\nCliquez sur 'Démarrer' pour relancer")
        self.emotion_display.setText("Aucune")
        self.confidence_display.setText("Confiance: 0.0%")
        self.add_log("⏹️ Détection arrêtée")

    def update_image(self, frame):
        """Met à jour l'image affichée"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
        except Exception as e:
            self.add_log(f"⚠️ update_image error: {e}")

    def update_emotion(self, emotion, confidence):
        """Met à jour l'affichage de l'émotion et son image associée"""
        self.emotion_display.setText(emotion)
        self.confidence_display.setText(f"Confiance: {confidence:.1f}%")

        # safe filename mapping (remove spaces/accents/emojis)
        safe_name = "".join(c for c in emotion.lower() if c.isalnum())
        if not safe_name:
            safe_name = emotion.lower().split()[0] if emotion else "unknown"

        emotion_image_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../app/assets/emotions", f"{safe_name}.png")
        )

        if os.path.exists(emotion_image_path):
            pixmap = QPixmap(emotion_image_path).scaled(
                self.emotion_image.width(),
                self.emotion_image.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.emotion_image.setPixmap(pixmap)
        else:
            self.emotion_image.clear()
            self.emotion_image.setText("📸 Aucune image")

    def change_camera(self, index):
        """Change la caméra utilisée"""
        self.camera_index = index
        self.add_log(f"📷 Caméra sélectionnée: {index}")

    def update_confidence_threshold(self, value):
        """Met à jour le seuil de confiance"""
        self.min_confidence = value / 100.0
        self.confidence_value_label.setText(f"{value}%")
        if self.video_thread:
            self.video_thread.update_confidence(value)

    def closeEvent(self, event):
        """Gère la fermeture de la fenêtre"""
        if self.video_thread:
            try:
                self.video_thread.stop()
            except Exception:
                pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = EmotionDetectionGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
