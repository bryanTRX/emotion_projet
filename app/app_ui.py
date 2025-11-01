import sys
import cv2
import random
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
    QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import QTimer, Qt


# === Placeholder emotion model ===
def detect_emotion_from_frame(frame):
    emotions = [
        ("Joy", "😄", "#FFD166"),
        ("Sadness", "😢", "#118AB2"),
        ("Anger", "😡", "#EF476F"),
        ("Fear", "😨", "#073B4C"),
        ("Disgust", "🤢", "#06D6A0")
    ]
    emotion, emoji, color = random.choice(emotions)
    confidence = random.uniform(80, 99)
    return emotion, emoji, confidence, color


class EmotionUI(QWidget):
    def __init__(self):
        super().__init__()

        # === Window setup ===
        self.setWindowTitle("Emotion Vision – Inside Out Mode")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("background-color: #0B132B; color: #E0E6F8;")

        # === Camera setup ===
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not accessible")

        # === Main layout ===
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # --- Video feed ---
        self.video_label = QLabel()
        self.video_label.setFrameShape(QFrame.Box)
        self.video_label.setStyleSheet("""
            border-radius: 15px;
            border: 3px solid #1C2541;
            background-color: #1C2541;
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.video_label, stretch=1)

        # --- Bottom Info Bar ---
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #1C2541;
                border-radius: 15px;
            }
        """)
        info_layout = QHBoxLayout()
        info_layout.setContentsMargins(20, 10, 20, 10)
        info_layout.setSpacing(40)

        self.emotion_label = QLabel("Emotion: --")
        self.emoji_label = QLabel("")
        self.conf_label = QLabel("Confidence: --")
        self.face_label = QLabel("Faces: 0")
        self.fps_label = QLabel("FPS: --")

        for lbl in [self.emotion_label, self.emoji_label, self.conf_label, self.face_label, self.fps_label]:
            lbl.setFont(QFont("Helvetica", 16, QFont.Bold))
            lbl.setStyleSheet("color: #E0E6F8;")

        info_layout.addWidget(self.emoji_label)
        info_layout.addWidget(self.emotion_label)
        info_layout.addWidget(self.conf_label)
        info_layout.addStretch()
        info_layout.addWidget(self.face_label)
        info_layout.addWidget(self.fps_label)

        info_frame.setLayout(info_layout)
        main_layout.addWidget(info_frame)

        self.setLayout(main_layout)

        # === Timer for updating camera ===
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.prev_time = time.time()
        self.smooth_fps = 0

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # FPS Calculation
        now = time.time()
        dt = now - self.prev_time if now != self.prev_time else 1e-6
        fps = 1.0 / dt
        self.smooth_fps = 0.9 * self.smooth_fps + 0.1 * fps if self.smooth_fps else fps
        self.prev_time = now

        # Detect emotion (placeholder)
        emotion, emoji, conf, color = detect_emotion_from_frame(frame)

        # Display emotion info
        self.emoji_label.setText(emoji)
        self.emotion_label.setText(f"Emotion: {emotion}")
        self.conf_label.setText(f"Confidence: {conf:.1f}%")
        self.fps_label.setText(f"FPS: {self.smooth_fps:.1f}")
        self.face_label.setText("Faces: 1")  # Replace with real face count later

        # Change emotion color dynamically
        self.emotion_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.emoji_label.setStyleSheet(f"color: {color}; font-size: 22px;")

        # Convert frame for Qt display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = EmotionUI()
    ui.show()
    sys.exit(app.exec_())
