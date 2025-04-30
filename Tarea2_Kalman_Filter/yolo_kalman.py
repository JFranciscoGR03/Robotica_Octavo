"""Seguimiento de aviones con YOLOv8 y Filtro de Kalman.

Este script realiza la detección y el seguimiento de aviones en un video.
Utiliza el modelo YOLOv8 para la detección y un Filtro de Kalman para el
seguimiento de múltiples objetos. A cada avión detectado se le asigna un ID
único y se rastrea a lo largo de los cuadros usando IoU (Intersección sobre
Unión) para asociar las detecciones con los rastreadores.
"""

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO


# Clase que representa un objeto seguido por un filtro de Kalman
class KalmanTracker:
    """
    Representa un objeto seguido por un filtro de Kalman.

    Cada instancia rastrea un objeto usando un modelo de movimiento constante
    y actualiza su estado con nuevas detecciones a lo largo del tiempo.
    """

    count = 0  # Contador estático para asignar ID único a cada tracker

    def __init__(self, bbox):
        """
        Inicializa el filtro de Kalman para seguir un objeto.

        Usa un estado de 8 dimensiones (posición y velocidad).
        """
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # Matriz de transición del estado (modelo de movimiento constante)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # Matriz de observación: solo observamos la posición (no velocidad)
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        # Ajuste de parámetros del filtro
        self.kf.R *= 10  # Ruido de medición
        self.kf.P *= 1000  # Incertidumbre inicial
        self.kf.Q *= 0.01  # Ruido del modelo

        # Inicializar el estado con la bounding box recibida
        x1, y1, x2, y2 = bbox
        self.kf.x[:4] = np.array([[x1], [y1], [x2], [y2]])

        # Asignar ID único al tracker
        self.id = KalmanTracker.count
        KalmanTracker.count += 1

        self.time_since_update = 0  # Contador de frames sin actualización

    def update(self, bbox):
        """Actualiza el filtro con una nueva observación (bounding box)."""
        x1, y1, x2, y2 = bbox
        self.kf.update(np.array([[x1], [y1], [x2], [y2]]))
        self.time_since_update = 0

    def predict(self):
        """Predice la siguiente posición del objeto."""
        self.kf.predict()
        self.time_since_update += 1
        x1, y1, x2, y2 = self.kf.x[:4].flatten()
        return int(x1), int(y1), int(x2), int(y2)


def iou(bbox1, bbox2):
    """Calcula el Intersection over Union entre dos cajas delimitadoras."""
    x1, y1, x2, y2 = bbox1
    x1_p, y1_p, x2_p, y2_p = bbox2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    bbox1_area = max(0, x2 - x1) * max(0, y2 - y1)
    bbox2_area = max(0, x2_p - x1_p) * max(0, y2_p - y1_p)

    union = bbox1_area + bbox2_area - inter_area
    return inter_area / union if union > 0 else 0


# Cargar el modelo YOLOv8 preentrenado
model = YOLO("/home/francisco/robotica_octavo_hsi/tarea2/yolov8m.pt")

# Abrir archivo de video
cap = cv2.VideoCapture(
    "/home/francisco/robotica_octavo_hsi/Tarea2_Kalman_Filter"
    "/Videos_Prueba/aviones_3.mp4"
)

trackers = []  # Lista de trackers activos

# Bucle principal de procesamiento
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = []  # Lista de bounding boxes detectadas
    results = model(frame)[0]  # Inferencia con YOLO

    # Filtrar las detecciones por clase y confianza
    for r in results.boxes:
        class_id = int(r.cls[0])
        conf = float(r.conf[0])
        if conf < 0.5 or class_id != 4:  # Clase 4 = avión
            continue

        x1, y1, x2, y2 = map(int, r.xyxy[0])
        detections.append((x1, y1, x2, y2))

    assigned = set()  # IDs de trackers ya actualizados en este frame

    # Asociar detecciones a trackers existentes
    for det in detections:
        best_iou = 0
        best_tracker = None
        for tracker in trackers:
            pred_bbox = tracker.predict()
            i = iou(pred_bbox, det)
            if i > best_iou:
                best_iou = i
                best_tracker = tracker

        if best_iou > 0.3 and best_tracker:
            best_tracker.update(det)
            assigned.add(best_tracker.id)
        else:
            # Si no hay coincidencia, crear un nuevo tracker
            new_tracker = KalmanTracker(det)
            trackers.append(new_tracker)

    # Dibujar los trackers en pantalla
    for tracker in trackers:
        x1, y1, x2, y2 = tracker.predict()
        if tracker.time_since_update < 10:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"ID {tracker.id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    # Mostrar el resultado en una ventana
    cv2.imshow("Seguimiento de Aviones", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos al finalizar
cap.release()
cv2.destroyAllWindows()
