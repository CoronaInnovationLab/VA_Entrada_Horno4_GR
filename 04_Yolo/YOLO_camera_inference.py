from utils import conectar_camara, draw_detections, draw_grids, show_inventary, preparar_img, preparar_img_raw, Tracker, save_sql, generar_alarma, detener_alarma, log
from ultralytics import YOLO
import numpy as np
import cv2 as cv
import torch
import time
import os

# ******************************************************
# Configuraciones iniciales
# ******************************************************
model_path: str = "runs/train/Entrada_H4_YOLO_SEG_X_V2_384x288/weights/best.pt" 
# model_path: str = "runs/train/best.pt" 
crop_path: str = "../00_Data/crops"

video_path: str = '../00_Data/videos'
output_path: str = '../00_Data/videos/inferencias'
video_iniciado: bool = False
carro_completo: bool = False

if not os.path.exists(output_path):
    os.makedirs(output_path)

# ******************************************************
# Configuracion camara y guardado de video
# ******************************************************
# Inicializar Harvester
h, ia = conectar_camara('entrada')

# ******************************************************
# Parametros del modelo
# ******************************************************
model = YOLO(model_path)
class_names = model.names
min_confidence: float  = 0.90
min_iou: float = 0.45

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

while True:
    #Preguntamos si no se esta dando la confirmacion de movimiento carro
    señal = ia.remote_device.node_map.LineStatus.value

    if not señal:
        carro_completo = False

    if señal and not carro_completo:

        # inicializar variables y captura de datos en la camara
        if not video_iniciado:

            alarma_choque = False
            tracker = Tracker()
            video_name: str = time.strftime("%Y-%m-%d_%H-%M-%S")

            output_inference_path = os.path.join(output_path, f"{video_name}_inference.mp4")
            output_raw_path = os.path.join(video_path, f"{video_name}_raw.mp4")

            fps = 10
            fourcc = 0x00000021#cv.VideoWriter_fourcc(*'x264')
            fourccraw = 0x00000021#cv.VideoWriter_fourcc(*'mpv4')

            inference = cv.VideoWriter(output_inference_path, fourcc, fps, (768, 576))
            raw = cv.VideoWriter(output_raw_path, fourccraw, fps, (2464, 2056))

            video_iniciado = True

            # Inicio captura de datos en camara
            ia.start()

        # Capturar una imagen de la cámara
        with ia.fetch() as buffer:
            # Obtener los datos del buffer
            component = buffer.payload.components[0]

            # Convertir los datos a una imagen numpy
            img_original_camara = component.data.reshape(component.height, component.width)

            # Crear una copia de la imagen
            image = img_original_camara.copy()

        # Preprocesar la imagen
        frame, roi = preparar_img(image)
        image = preparar_img_raw(image)

        # Inferir
        results = model.predict(source=roi, conf=min_confidence, iou=min_iou, device=device, verbose=False)
        
        # Procesar resultados
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            #convertir labels de float a su equivalente en string
            labels = np.array([class_names[int(label)] for label in labels])
            confidences = result.boxes.conf.cpu().numpy()

            # Actualizar el inventario
            carro_completo = tracker.update(boxes, labels)
            
            alarma_choque = tracker.contar(frame, crop_path) or alarma_choque

            # mostrar el inventario
            show_inventary(frame, tracker.inventario)

            # Dibujar detecciones
            for box, label, confidence in zip(boxes, labels, confidences):
                frame = draw_detections(frame, box, label, confidence)

        #mostrar bordes
        frame = draw_grids(frame)

        # Guardar fotogramas
        raw.write(image)
        inference.write(frame)

        # Ver en tiempo real
        cv.imshow("YOLO Inference", frame)

        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv.imwrite('captura_entrada.png', frame)

        if carro_completo:
            
            ia.stop()

            inference.release()
            inference = None

            raw.release()
            raw = None
            
            video_iniciado = False
            cv.destroyAllWindows()
            #
            save_sql(tracker.inventario, video_name, alarma_choque)
            if alarma_choque:
                generar_alarma()
            log(tracker.inventario)

            # sleep para que suene la alarma
            time.sleep(5)
            if alarma_choque:
                detener_alarma()