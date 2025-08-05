import cv2

# Ruta del video original
input_video = "../00_Data/video_oscuro.mp4"
# Ruta donde se guardará el video mejorado
output_video = "../00_Data/video_brillo_contraste.mp4"

# Parámetros para mejorar brillo y contraste
alpha = 1.8  # Contraste (>1 sube contraste, entre [0-1] lo baja)
beta = 60    # Brillo (positivo aclara, negativo oscurece)

# Abrir el video original
cap = cv2.VideoCapture(input_video)

# Leer ancho, alto y FPS del video
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Definir códec y creador de video de salida
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Aumentar brillo y contraste
    # dst = alpha * frame + beta
    frame_enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Escribir en el archivo de video
    out.write(frame_enhanced)

# Liberar recursos
cap.release()
out.release()

print("Proceso finalizado. Video guardado en:", output_video)
