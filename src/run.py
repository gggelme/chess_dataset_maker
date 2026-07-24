import cv2 as cv
import numpy as np
import chess
import os
import sys

# python src/run.py

dir_src = os.path.dirname(os.path.abspath(__file__)) 
dir_raiz = os.path.dirname(dir_src)                  

sys.path.insert(0, os.path.join(dir_src, 'parser'))
sys.path.insert(0, os.path.join(dir_src, 'ui'))
sys.path.insert(0, dir_src)

from detect_movements import obtener_celdas_cambiadas, inferir_movimiento, chess_board_a_matriz
from parser_table import DetectorTablero
from virtual_board import LiveBoard
from stockfish_advisor import StockfishAdvisor
from data_logger import GameLogger


# ── Configuración ─────────────────────────────────────────────────────────────
VIVO = True
URL = os.path.join(dir_raiz, "data", "raw", "partida_larga_normal.mp4")
UMBRAL_PIEZA = 0.15  # 15% de área de celda alterada
OFFSET_TABLERO = 70  # Ajuste para descartar bordes físicos


def main():
    cap = cv.VideoCapture(1 if VIVO else URL)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir la cámara o el archivo.")
        return
        
    if VIVO:
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)

    # Inicialización de lógica y visión
    parser = DetectorTablero(offset=OFFSET_TABLERO)
    board_logico = chess.Board()
    prev_corners = None
    warp_ref = None

    # Inicialización de UI y Datos
    live_board = LiveBoard()
    live_board.actualizar(chess_board_a_matriz(board_logico))
    
    logger = GameLogger(os.path.join(dir_raiz, "data", "log"))
    
    try:
        advisor = StockfishAdvisor()
        print("Stockfish listo.")
    except Exception as e:
        advisor = None
        print(f"Stockfish no disponible: {e}")

    print("Tablero inicializado. Presiona ESPACIO para inferir jugada o 'R' para resetear referencia.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Refresco constante de la interfaz gráfica (PyGame u otra UI)
        if live_board is not None:
            if advisor is not None:
                live_board.sugerencias = advisor.get_sugerencias()
            if not live_board.actualizar(chess_board_a_matriz(board_logico)):
                break # Si cierran la ventana del tablero virtual, cortamos

        # 2. Procesamiento visual base
        parser.update_frame(frame)
        try:
            prev_corners = parser.detect_board_corners(prev_corners)
            tablero_bgr = parser.get_board_roi()
        except ValueError:
            cv.imshow("Chess Vision", cv.resize(frame, (0, 0), fx=0.5, fy=0.5))
            if cv.waitKey(30) & 0xFF == 27: break
            continue

        tablero_gray = cv.cvtColor(tablero_bgr, cv.COLOR_BGR2GRAY)
        tablero_gray = cv.GaussianBlur(tablero_gray, (5, 5), 0)

        if warp_ref is None:
            warp_ref = tablero_gray.copy()
            continue

        # 3. Mostrar ventana de debug en vivo
        cv.imshow("Chess Vision", cv.resize(frame, (0, 0), fx=0.5, fy=0.5))
        diff_preview = cv.absdiff(tablero_gray, warp_ref)

        # 4. Captura de eventos discretos
        key = cv.waitKey(30) & 0xFF
        
        if key == 27 or key == ord('q'): # Salir
            break
            
        elif key == ord('r') or key == ord('R'): # Reseteo manual
            warp_ref = tablero_gray.copy()
            print("\n[!] Referencia visual limpia.")
            
        elif key == ord(' '): # EVENTO: Analizar jugada
            # 1. Binarizamos para aislar el área que cambió
            _, diff_thresh = cv.threshold(diff_preview, 25, 255, cv.THRESH_BINARY)
            diff_clean = cv.morphologyEx(diff_thresh, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
            
            # 2. Pasamos diff_clean y el umbral porcentual (0.15) correcto
            cambiadas, energias_celdas = obtener_celdas_cambiadas(diff_clean, UMBRAL_PIEZA)
            print(f"\n[*] INTERRUPCIÓN MANUAL. Analizando celdas: {cambiadas}")

            # Capturar estado antes de la jugada
            turno_antes = board_logico.turn
            clave_sug = 'blancas' if turno_antes == chess.WHITE else 'negras'
            datos_sug = live_board.sugerencias.get(clave_sug) if live_board else None
            
            # 3. Desempaquetamos la tupla (texto, uci) de forma segura
            sug_txt = datos_sug[0] if isinstance(datos_sug, tuple) else datos_sug

            # Inferencia lógica
            mov, san = inferir_movimiento(board_logico, cambiadas, energias_celdas)

            if mov:
                print(f"[+] Movimiento Legal Validado: {mov.uci()} ({san})")
                logger.registrar(turno_antes, san, mov.uci(), sug_txt)
                
                if board_logico.is_game_over():
                    resultado = board_logico.result()
                    live_board.set_resultado(resultado)
                    live_board.sugerencias = {'blancas': None, 'negras': None} 
                    print(f"\n[*] PARTIDA FINALIZADA. Resultado: {resultado}")
                else:
                    if advisor is not None:
                        advisor.analizar_async(board_logico)
                
                live_board.actualizar(chess_board_a_matriz(board_logico))
            else:
                print("[-] Movimiento INVÁLIDO. Revisá el tablero físico.")

            warp_ref = tablero_gray.copy()

    # Cierre de módulos
    logger.guardar_historial_completo()
    if advisor is not None:
        advisor.cerrar()

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()