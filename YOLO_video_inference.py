from utils import draw_detections, draw_grids, show_inventary, preparar_img, Tracker
from ultralytics import YOLO
import numpy as np
import cv2 as cv
import torch
import os

model_path: str = "runs/train/Entrada_H4_YOLO_V1/weights/best.pt" 
crop_path: str = "data/crops"
video_name: str = "crudo_2025-01-10_13-20-38.mp4"

video_path: str = os.path.join('data/videos', video_name)
output_path: str = 'data/videos/inferencias'
save_video: bool = True

cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file")
    exit()
if not os.path.exists(output_path):
    os.makedirs(output_path)
if save_video:
    output_path = os.path.join(output_path, f"{video_name[:-4]}_inference.mp4")
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(768)
    height = int(576)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

model = YOLO(model_path)
class_names = model.names
min_confidence: float  = 0.90
min_iou: float = 0.45
alarma_choque = False

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Inicializar tracker
tracker = Tracker()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesar la imagen
    frame, mascara = preparar_img(frame)

    # Inferir
    results = model.predict(source=mascara, conf=min_confidence, iou=min_iou, device=device)
    
    # Procesar resultados
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()
        #convertir labels de float a su equivalente en string
        labels = np.array([class_names[int(label)] for label in labels])
        confidences = result.boxes.conf.cpu().numpy()

        # Actualizar el inventario
        fin = tracker.update(boxes, labels)
        print(f"Total objects: {len(tracker.tracks)}")

        alarma_choque = tracker.contar(frame, crop_path) or alarma_choque

        # mostrar el inventario
        show_inventary(frame, tracker.inventario)

        # Dibujar detecciones
        for box, label, confidence in zip(boxes, labels, confidences):
            frame = draw_detections(frame, box, label, confidence)

    if fin:
        break

    #mostrar bordes
    frame = draw_grids(frame)

    # Ver en tiempo real
    cv.imshow("YOLOv8 Inference", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
        
    if save_video:
        out.write(frame)

print(f"Alarma de choque: {alarma_choque}")
cap.release()
out.release()
cv.destroyAllWindows()
