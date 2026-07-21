import cv2
import numpy as np
import os
import sys

# python src/parser/parser_table_debug.py

dir_actual = os.path.dirname(os.path.abspath(__file__))
dir_proyecto = os.path.dirname(os.path.dirname(dir_actual))
sys.path.insert(0, dir_proyecto)

from src.parser.parser_table import ParserTable

def test_parser_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return

    # Color amarillo claro en BGR
    AMARILLO_CLARO = (153, 255, 255) 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        visor = frame.copy()

        try:
            # 1. Ejecutar el pipeline de detección
            parser = ParserTable(gris)
            parser.detect_board_corners()
            parser.correct_perspective(800)
            parser.standardize_orientation()
            y_pos, x_pos = parser.detect_grid_lines()

            # 2. Dibujar el contorno del tablero (Esquinas originales)
            cv2.polylines(visor, [parser.esquinas], isClosed=True, color=AMARILLO_CLARO, thickness=3)

            # 3. Dibujar la grilla proyectada
            # Creamos un lienzo negro del tamaño del tablero rectificado
            grilla_rectificada = np.zeros((800, 800, 3), dtype=np.uint8)

            for x in x_pos:
                cv2.line(grilla_rectificada, (x, 0), (x, 800), AMARILLO_CLARO, 2)
            for y in y_pos:
                cv2.line(grilla_rectificada, (0, y), (800, y), AMARILLO_CLARO, 2)

            # Aplicamos la homografía inversa para llevar la grilla al espacio de la cámara
            grilla_original = cv2.warpPerspective(grilla_rectificada, parser.H, (frame.shape[1], frame.shape[0]), flags=cv2.WARP_INVERSE_MAP)

            # Sumamos las imágenes para superponer la grilla sobre el frame original
            visor = cv2.addWeighted(visor, 1.0, grilla_original, 0.8, 0)

        except Exception as e:
            # Si un frame sale movido o borroso y falla el parser, lo mostramos en rojo
            cv2.putText(visor, f"Fallo Deteccion: {e}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Mostrar el resultado
        cv2.imshow("Debug - ParserTable", visor)

        # Controles: 'q' para salir, 'espacio' para pausar frame por frame
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ruta_video = os.path.join(dir_proyecto, "data", "raw", "partida_larga_normal.mp4")
    test_parser_video(ruta_video)