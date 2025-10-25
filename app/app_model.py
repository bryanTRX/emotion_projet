import cv2
import mediapipe as mp
import time
import math
import sys
import signal
from typing import Optional, Tuple


def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness=2, radius=20, fill=False):
    x1, y1 = top_left
    x2, y2 = bottom_right
    w = x2 - x1
    h = y2 - y1
    radius = max(0, min(radius, int(min(w, h) / 2)))

    if fill:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, cv2.FILLED)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, cv2.FILLED)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, cv2.FILLED)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, cv2.FILLED)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, cv2.FILLED)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, cv2.FILLED)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def init_camera(camera_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> cv2.VideoCapture:
    """
    Initialise et retourne un objet VideoCapture.
    Lève RuntimeError si la caméra ne peut pas être ouverte.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'ouvrir la caméra (index={camera_index}).")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def try_reconnect(camera_index: int, retries: int = 3, wait_seconds: float = 1.0) -> Optional[cv2.VideoCapture]:
    """
    Essaie de rouvrir la caméra 'retries' fois, avec 'wait_seconds' entre chaque tentative.
    Retourne un VideoCapture ouvert ou None.
    """
    for attempt in range(1, retries + 1):
        try:
            print(f"Tentative de reconnexion à la caméra (essai {attempt}/{retries})...")
            cap = init_camera(camera_index)
            print("Reconnexion OK.")
            return cap
        except RuntimeError as e:
            print(f"Échec de reconnexion: {e}")
            time.sleep(wait_seconds)
    print("Reconnexion échouée après toutes les tentatives.")
    return None


def draw_fps(frame, fps_value: float, font_scale=0.6, pos: Tuple[int,int]=None):
    text = f"FPS: {fps_value:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    if pos is None:
        x = frame.shape[1] - text_size[0] - 10
        y = 10 + text_size[1]
    else:
        x, y = pos
    cv2.rectangle(frame, (x - 6, y - text_size[1] - 6), (x + text_size[0] + 6, y + 6), (0,0,0), cv2.FILLED)
    cv2.putText(frame, text, (x, y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

def draw_face_count(frame, count: int, font_scale=0.6):
    text = f"Visages : {count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    cv2.rectangle(frame, (6, 6), (6 + text_size[0] + 12, 6 + text_size[1] + 12), (0,0,0), cv2.FILLED)
    cv2.putText(frame, text, (12, 6 + text_size[1] + 2), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)


def detect_face_mediapipe(camera_index=0,
                          max_num_faces=3,
                          reconnect_retries=3,
                          reconnect_wait=1.0,
                          face_box_alpha=0.25):
    """
    Version propre et structurée :
    - gère les erreurs d'ouverture et de lecture de la caméra,
    - tente de reconnexion si le flux est perdu,
    - nettoie proprement ressources et fenêtres sur sortie.
    """
    face_box_color = (0, 200, 255)
    contour_color = (0, 255, 0)
    corner_radius = 20
    margin_rel = 0.12

    prev_time = time.time()
    fps = 0.0
    smoothing = 0.85

    stop_flag = {"stop": False}
    def _signal_handler(sig, frame):
        print("Signal reçu, fermeture...")
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    cap = None
    try:
        cap = init_camera(camera_index)
        print("Caméra ouverte. Appuyez sur 'q' ou Ctrl+C pour quitter.")

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh

        with mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            while True:
                if stop_flag["stop"]:
                    print("Arrêt demandé.")
                    break

                ret, frame = cap.read()
                if not ret:
                    print("Lecture du flux vidéo échouée.")
                    cap.release()
                    cap = try_reconnect(camera_index, retries=reconnect_retries, wait_seconds=reconnect_wait)
                    if cap is None:
                        print("Impossible de récupérer le flux. Sortie.")
                        break
                    else:
                        continue

                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                frame.flags.writeable = True

                overlay = frame.copy()
                face_count = 0

                if results.multi_face_landmarks:
                    face_count = len(results.multi_face_landmarks)
                    for i, face_landmarks in enumerate(results.multi_face_landmarks):
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                        coords = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                        xs, ys = zip(*coords)
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)

                        box_w = x_max - x_min
                        box_h = y_max - y_min
                        pad_x = int(box_w * margin_rel)
                        pad_y = int(box_h * margin_rel)

                        x1 = max(0, x_min - pad_x)
                        y1 = max(0, y_min - pad_y)
                        x2 = min(w - 1, x_max + pad_x)
                        y2 = min(h - 1, y_max + pad_y)

                        dynamic_color = (
                            int(face_box_color[0] * (1 - i*0.1)) % 256,
                            int(face_box_color[1] * (1 - i*0.1)) % 256,
                            int(face_box_color[2] * (1 - i*0.1)) % 256
                        )

                        draw_rounded_rectangle(overlay, (x1, y1), (x2, y2), dynamic_color, thickness=1, radius=corner_radius, fill=True)
                        draw_rounded_rectangle(frame, (x1, y1), (x2, y2), contour_color, thickness=2, radius=corner_radius, fill=False)

                        label = f"Visage #{i+1}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        text_thickness = 2
                        tx = x1
                        ty = max(20, y1 - 10)
                        rect_x1 = tx - 4
                        rect_y1 = ty - 18
                        rect_x2 = tx + 90
                        rect_y2 = ty + 6
                        draw_rounded_rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), dynamic_color, thickness=1, radius=8, fill=True)
                        cv2.putText(frame, label, (tx, ty), font, font_scale, (255,255,255), text_thickness, cv2.LINE_AA)

                cv2.addWeighted(overlay, face_box_alpha, frame, 1 - face_box_alpha, 0, frame)

                current_time = time.time()
                dt = current_time - prev_time if current_time != prev_time else 1e-6
                instant_fps = 1.0 / dt
                fps = smoothing * fps + (1 - smoothing) * instant_fps if fps != 0 else instant_fps
                prev_time = current_time

                draw_fps(frame, fps)
                draw_face_count(frame, face_count)

                cv2.imshow("Détection de visage - Propre & fiable", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Sortie demandée par l'utilisateur (q).")
                    break

    except RuntimeError as e:
        print(f"Erreur d'exécution : {e}", file=sys.stderr)
    except Exception as e:
        print("Exception inattendue :", e, file=sys.stderr)
    finally:
        try:
            if cap is not None and cap.isOpened():
                cap.release()
                print("Camera released.")
        except Exception as e:
            print("Erreur lors du release de la caméra :", e, file=sys.stderr)
        try:
            cv2.destroyAllWindows()
            print("Toutes les fenêtres fermées.")
        except Exception as e:
            print("Erreur lors de la destruction des fenêtres OpenCV :", e, file=sys.stderr)


if __name__ == "__main__":
    detect_face_mediapipe(camera_index=0, max_num_faces=3, reconnect_retries=3, reconnect_wait=1.0)
