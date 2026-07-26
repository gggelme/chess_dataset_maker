import cv2 as cv
import numpy as np
import chess
import os
import sys

# python src/run.py

# Rutas
dir_src = os.path.dirname(os.path.abspath(__file__)) 
dir_raiz = os.path.dirname(dir_src)                  
sys.path.insert(0, os.path.join(dir_src, 'parser'))
sys.path.insert(0, os.path.join(dir_src, 'ui'))
sys.path.insert(0, dir_src)

from movement_detection import DetectorMovimiento, chess_board_a_matriz
from parser_table import DetectorTablero, configurar_offset
from virtual_board import LiveBoard
from stockfish_advisor import StockfishAdvisor
from menu import iniciar_menu
from viewer import visualizar_partida
from data_logger import GameLogger

# ── Configuración ─────────────────────────────────────────────────────────────
VIVO = True
URL = os.path.join(dir_raiz, "data", "raw", "Prueba2.mp4")  

def iniciar_deteccion():
    cap = cv.VideoCapture(1 if VIVO else URL)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara o el archivo de video.")
        return
        
    if VIVO:
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)

    offset_elegido = configurar_offset(cap, 0)

    if not VIVO:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    # Inicializamos los nuevos módulos de visión
    parser = DetectorTablero(offset=offset_elegido)
    detector_mov = DetectorMovimiento(umbral_energia=11.0, frames_estabilidad=15)
    prev_corners = None

    # Inicialización de UI y Datos (usando el board de la clase)
    live_board = LiveBoard()
    live_board.actualizar(chess_board_a_matriz(detector_mov.board_logico))
    logger = GameLogger(os.path.join(dir_raiz, "data", "log"))
    
    try: advisor = StockfishAdvisor()
    except Exception: advisor = None

    print("\n[*] Iniciando detección de movimientos. Presiona 'q' o 'Esc' para salir, 'r' para resetear referencia de luz.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue

        # 1. Refresco de UI con el tablero lógico encapsulado
        if live_board is not None:
            if advisor is not None:
                live_board.sugerencias = advisor.get_sugerencias()
            if not live_board.actualizar(chess_board_a_matriz(detector_mov.board_logico)): break

        # 2. Extracción del ROI
        parser.update_frame(frame)
        try:
            prev_corners = parser.detect_board_corners(prev_corners)
            tablero_bgr = parser.get_board_roi()
        except ValueError:
            cv.imshow("Chess Vision", cv.resize(frame, (0, 0), fx=0.5, fy=0.5))
            if cv.waitKey(30) & 0xFF == 27: break
            continue

        # 3. Detección automática de movimientos
        mov, san = detector_mov.procesar_roi(tablero_bgr)

        if mov:
            # Como la clase ya hizo el push() del movimiento, el turno anterior es el inverso
            turno_antes = not detector_mov.board_logico.turn 
            clave_sug = 'blancas' if turno_antes == chess.WHITE else 'negras'
            datos_sug = live_board.sugerencias.get(clave_sug) if live_board else None
            sug_txt = datos_sug[0] if isinstance(datos_sug, tuple) else datos_sug

            logger.registrar(turno_antes, san, mov.uci(), sug_txt)
            
            if detector_mov.board_logico.is_game_over():
                live_board.set_resultado(detector_mov.board_logico.result())
                live_board.sugerencias = {'blancas': None, 'negras': None} 
                print(f"\n[*] PARTIDA FINALIZADA.")
            elif advisor is not None:
                advisor.analizar_async(detector_mov.board_logico)

        cv.imshow("Chess Vision", cv.resize(frame, (0, 0), fx=0.5, fy=0.5))

        # 4. Control de eventos
        key = cv.waitKey(30) & 0xFF
        if key in [27, ord('q')]: 
            break
        elif key in [ord('r'), ord('R')]: 
            detector_mov.referencia_medias = None
            detector_mov.buffer_medias.clear()
            print("\n[!] Referencia visual y buffers limpios.")

    # Cierre
    logger.guardar_historial_completo()
    if advisor is not None: advisor.cerrar()
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    iniciar_menu(iniciar_deteccion, visualizar_partida)