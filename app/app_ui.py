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
    
    def __init__(self, model_path, camera_index, min_confidence):
        super().__init__()
        self.model_path = model_path
        self.camera_index = camera_index
        self.min_confidence = min_confidence
        self.running = False
        self.paused = False
        
        
    def log(self, message):
        """Envoie un message de log à l'interface"""
        print(message, flush=True)  # Force l'affichage dans le terminal
        self.log_signal.emit(message)
        
    def run(self):
        """Boucle principale du thread vidéo"""
        try:
            self.log("🔄 Chargement du modèle...")
            self.model, self.device = load_model(self.model_path)
            self.model.eval()
            self.log("✅ Modèle chargé avec succès")
            
            self.log("🔄 Initialisation de MediaPipe...")
            mp_face = mp.solutions.face_detection
            self.detector = mp_face.FaceDetection(
                model_selection=0, 
                min_detection_confidence=self.min_confidence
            )
            self.log("✅ MediaPipe initialisé")
            
            self.log(f"🔄 Tentative d'ouverture de la caméra {self.camera_index}...")
            self.cap = cv2.VideoCapture(self.camera_index)
            
            # Attendre l'initialisation de la caméra
            import time
            time.sleep(1)
            
            if not self.cap.isOpened():
                error_msg = f"❌ Impossible d'ouvrir la caméra {self.camera_index}"
                self.log(error_msg)
                self.error_signal.emit("Caméra non accessible. Vérifiez:\n1. Les permissions système\n2. Qu'aucune autre app n'utilise la caméra\n3. Essayez une autre caméra")
                return
            
            # Test de lecture
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                error_msg = "❌ Caméra ouverte mais impossible de lire les frames"
                self.log(error_msg)
                self.error_signal.emit("Erreur de lecture vidéo")
                self.cap.release()
                return
                
            self.log(f"✅ Caméra opérationnelle (résolution: {test_frame.shape[1]}x{test_frame.shape[0]})")
            self.running = True
            
            frame_count = 0
            while self.running:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if ret:
                        frame = self.process_frame(frame)
                        self.change_pixmap_signal.emit(frame)
                        frame_count += 1
                        if frame_count == 1:
                            self.log("✅ Flux vidéo actif")
                    else:
                        self.log("❌ Erreur de lecture frame")
                        break
                else:
                    self.msleep(100)
                    
            self.cap.release()
            self.log("✅ Caméra fermée proprement")
            
        except Exception as e:
            error_msg = f"❌ ERREUR: {str(e)}"
            self.log(error_msg)
            import traceback
            self.log(traceback.format_exc())
            self.error_signal.emit(f"Erreur critique:\n{str(e)}")
    
    def process_frame(self, frame):
        """Traite une frame et détecte les émotions"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        
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
                
                tensor = preprocess_image(face_crop).to(self.device)
                with torch.no_grad():
                    outputs = self.model(tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                    
                label = get_emotion_label(pred.item())
                confidence = conf.item() * 100
                
                # Émettre le signal avec l'émotion détectée
                self.emotion_signal.emit(label, confidence)
                
                # Dessiner sur la frame
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
        
        return frame
    
    def stop(self):
        """Arrête le thread"""
        self.log("🔄 Arrêt du thread...")
        self.running = False
        self.wait()
    
    def pause(self):
        """Met en pause le traitement"""
        self.paused = True
        self.log("⏸️ Détection en pause")
    
    def resume(self):
        """Reprend le traitement"""
        self.paused = False
        self.log("▶️ Détection reprise")
    
    def update_confidence(self, value):
        """Met à jour le seuil de confiance"""
        self.min_confidence = value / 100.0


class EmotionDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Détection d'Émotions en Temps Réel")
        self.setGeometry(100, 100, 1200, 750)
        self.setStyleSheet("""
    QMainWindow {
        background: qlineargradient(
            spread:pad, 
            x1:0, y1:0, x2:1, y2:1, 
            stop:0 #2E1A47, stop:0.5 #332B5E, stop:1 #1C0C3C
        );
    }
    QLabel {
        color: white;
        font-family: 'Comic Sans MS', 'Arial Rounded MT Bold';
    }
    QPushButton {
        background-color: qlineargradient(
            spread:pad, 
            x1:0, y1:0, x2:1, y2:1, 
            stop:0 #6A11CB, stop:1 #2575FC
        );
        color: white;
        padding: 10px;
        font-size: 15px;
        font-weight: bold;
        border-radius: 12px;
        border: 2px solid #FFFFFF33;
        transition: all 0.3s;
    }
    QPushButton:hover {
        background-color: qlineargradient(
            spread:pad, 
            x1:0, y1:0, x2:1, y2:1, 
            stop:0 #833AB4, stop:1 #FD1D1D
        );
        transform: scale(1.05);
    }
    QPushButton:pressed {
        background-color: #FF512F;
    }
    QPushButton:disabled {
        background-color: #666666;
        color: #cccccc;
    }
    QGroupBox {
        color: #FDFDFD;
        font-weight: bold;
        font-size: 15px;
        border: 2px solid #FFFFFF44;
        border-radius: 10px;
        margin-top: 10px;
        padding-top: 10px;
        background-color: rgba(255, 255, 255, 0.05);
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: #FFD700;
        font-size: 16px;
        text-shadow: 1px 1px 3px #000;
    }
    QTextEdit {
        background-color: rgba(10, 10, 10, 0.7);
        color: #00FFAA;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        border: 2px solid #00FFAA;
        border-radius: 8px;
    }
    QSlider::groove:horizontal {
        border: 1px solid #bbb;
        height: 10px;
        background: #444;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: qradialgradient(
            cx:0.5, cy:0.5, radius:0.8, 
            fx:0.5, fy:0.5, 
            stop:0 #FFD700, stop:1 #FFA500
        );
        border: 1px solid #fff;
        width: 18px;
        margin: -5px 0;
        border-radius: 9px;
    }
    QComboBox {
        background-color: #222;
        color: white;
        border: 1px solid #555;
        border-radius: 6px;
        padding: 5px;
    }
""")

        # Variables
        self.video_thread = None
        self.is_running = False
        self.current_emotion = "Aucune"
        self.current_confidence = 0.0
        
        # Configuration par défaut
        self.model_path = "../models/best_model_traced.pt"
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
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #666666;
            }
        """)
        control_layout.addWidget(self.stop_button)
        
        left_panel.addLayout(control_layout)
        
        # Console de logs
        log_label = QLabel("📋 Journal d'activité:")
        left_panel.addWidget(log_label)
        
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(100)
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
        self.emotion_image.setStyleSheet("background: transparent; border: none;")
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
        # Auto-scroll vers le bas
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def show_error(self, error_msg):
        """Affiche une erreur dans l'interface"""
        self.video_label.setText(f"❌ ERREUR\n\n{error_msg}\n\nVoir le journal ci-dessous")
        self.add_log(f"❌ ERREUR: {error_msg}")
        self.stop_detection()
    
    def start_detection(self):
        """Démarre la détection d'émotions"""
        if not self.is_running:
            self.add_log("=" * 50)
            self.add_log("▶️ DÉMARRAGE DE LA DÉTECTION")
            self.add_log("=" * 50)
            
            self.video_label.setText("🔄 Initialisation en cours...")
            
            self.video_thread = VideoThread(
                self.model_path,
                self.camera_index,
                self.min_confidence
            )
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.emotion_signal.connect(self.update_emotion)
            self.video_thread.log_signal.connect(self.add_log)
            self.video_thread.error_signal.connect(self.show_error)
            self.video_thread.start()
            
            self.is_running = True
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.camera_combo.setEnabled(False)
    
    def pause_detection(self):
        """Met en pause/reprend la détection"""
        if self.video_thread:
            if self.pause_button.text() == "⏸️ Pause":
                self.video_thread.pause()
                self.pause_button.setText("▶️ Reprendre")
            else:
                self.video_thread.resume()
                self.pause_button.setText("⏸️ Pause")
    
    def stop_detection(self):
        """Arrête la détection"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
            
        self.is_running = False
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("⏸️ Pause")
        self.stop_button.setEnabled(False)
        self.camera_combo.setEnabled(True)
        
        self.video_label.clear()
        self.video_label.setText("📹 Caméra arrêtée\n\nCliquez sur 'Démarrer' pour relancer")
        self.emotion_display.setText("Aucune")
        self.confidence_display.setText("Confiance: 0.0%")
        self.add_log("⏹️ Détection arrêtée")
    
    def update_image(self, frame):
        """Met à jour l'image affichée"""
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
    
    def update_emotion(self, emotion, confidence):
        """Met à jour l'affichage de l'émotion et son image associée"""
        self.emotion_display.setText(emotion)
        self.confidence_display.setText(f"Confiance: {confidence:.1f}%")

        self.emotion_display.setStyleSheet("""
            color: white;
            border-radius: 15px;
            padding: 20px;
        """)

        # --- Image associée à l'émotion ---
        emotion_image_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "../app/assets/emotions", f"{emotion.lower()}.png")
        )

        if os.path.exists(emotion_image_path):
            pixmap = QPixmap(emotion_image_path).scaled(
                self.emotion_image.width(),
                self.emotion_image.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.emotion_image.setPixmap(pixmap)
            self.emotion_image.setAlignment(Qt.AlignCenter)  # <-- important!
        else:
            self.emotion_image.clear()
            self.emotion_image.setAlignment(Qt.AlignCenter)
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
            self.video_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = EmotionDetectionGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()