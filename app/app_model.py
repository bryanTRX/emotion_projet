import cv2
import mediapipe as mp

def detect_face_mediapipe(camera_index=0):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la camera.")
        return
    print("Caméra ouverte. Appuyez sur 'q' pour quitter.")

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur de lecture du flux vidéo.")
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)

            cv2.imshow("Detection de visage (MediaPipe)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera fermee proprement.")


if __name__ == "__main__":
    detect_face_mediapipe()
