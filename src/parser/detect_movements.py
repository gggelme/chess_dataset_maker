import cv2 as cv
import numpy as np
import chess
import time
import os
import sys

# Ajuste de rutas
dir_actual = os.path.dirname(os.path.abspath(__file__))
dir_raiz = os.path.dirname(os.path.dirname(dir_actual))
sys.path.insert(0, dir_raiz)

from extra.parser_table_viejo import ParserTable
carpeta_data_raw = os.path.join(dir_raiz, "data", "raw")

def get_energia(imagen):
    """Media de cuadrados de píxeles para medir movimiento."""
    return np.mean(imagen.astype(np.float32) ** 2)

def inicializar_tablero(frame_gris, lado=800):
    """Detecta el tablero y devuelve el parser configurado."""
    parser = ParserTable(frame_gris)
    parser.detect_board_corners()
    parser.correct_perspective(lado)
    parser.standardize_orientation()
    parser.detect_grid_lines()
    print("Tablero inicializado correctamente.")
    return parser

def celda_a_square(fila, col):
    return chess.square(col, 7 - fila)

def celda_a_uci(fila, col):
    return "abcdefgh"[col] + str(8 - fila)

def obtener_celdas_cambiadas(ref_warp, nuevo_warp, parser, umbral_pieza, max_celdas=6):
    """Calcula las energías directamente sobre los recortes (warps)."""
    diff_warp = cv.absdiff(nuevo_warp, ref_warp)

    energias = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            y1, y2 = parser.y_pos[i], parser.y_pos[i + 1]
            x1, x2 = parser.x_pos[j], parser.x_pos[j + 1]
            energias[i, j] = get_energia(diff_warp[y1:y2, x1:x2])

    validos = np.where(energias.ravel() > umbral_pieza)[0]
    if not validos.size:
        return [], energias

    ordenados = validos[np.argsort(energias.ravel()[validos])[::-1]][:max_celdas]
    return [(int(idx // 8), int(idx % 8)) for idx in ordenados], energias

def _celdas_afectadas(board, mov):
    """Simula el movimiento y devuelve las celdas que cambian de estado."""
    _simbolos = lambda b: {sq: p.symbol() for sq, p in b.piece_map().items()}
    antes = _simbolos(board)
    board.push(mov)
    despues = _simbolos(board)
    board.pop()

    afectadas = set()
    for sq in set(antes) | set(despues):
        if antes.get(sq) != despues.get(sq):
            afectadas.add((7 - chess.square_rank(sq), chess.square_file(sq)))
    return afectadas

def inferir_movimiento_legal(board, cambiadas, energias):
    """Deduce el movimiento jugado utilizando las celdas con mayor energía."""
    if len(cambiadas) < 2: return None
    
    turno = board.turn
    set_cambiadas = set(cambiadas)
    
    origenes = [(f, c) for f, c in cambiadas if board.piece_at(celda_a_square(f, c)) 
                and board.piece_at(celda_a_square(f, c)).color == turno]

    if not origenes: return None

    candidatos = []
    for (fo, co) in origenes:
        for (fd, cd) in cambiadas:
            if (fd, cd) == (fo, co): continue
            
            base_uci = celda_a_uci(fo, co) + celda_a_uci(fd, cd)
            for sufijo in ("", "q", "r", "b", "n"):
                try: mov = chess.Move.from_uci(base_uci + sufijo)
                except ValueError: continue
                
                if mov in board.legal_moves:
                    match_score = len(_celdas_afectadas(board, mov) & set_cambiadas)
                    energia_total = float(energias[fo, co] + energias[fd, cd])
                    candidatos.append((match_score, energia_total, mov))
                    break

    if not candidatos: return None

    candidatos.sort(key=lambda x: (x[0], x[1]), reverse=True)
    mejor_mov = candidatos[0][2]
    
    board.push(mejor_mov)
    print(f"  OK -- {mejor_mov.uci()} | turno ahora: {'blanco' if board.turn == chess.WHITE else 'negro'}")
    return mejor_mov

def ver_por_frame(video_warps, energias, interrupciones, referencias_warps):
    """Visor interactivo que muestra las imágenes recortadas y alineadas."""
    if not video_warps: return

    def on_trackbar(x):
        warp = video_warps[x]
        ref = referencias_warps[x]
        diff = cv.absdiff(warp, ref)
        
        # Pasamos a BGR solo para poder imprimir texto en color
        warp_bgr = cv.cvtColor(warp, cv.COLOR_GRAY2BGR)
        color = (0, 0, 255) if interrupciones[x] else (0, 255, 0)
        
        cv.putText(warp_bgr, f"Frame: {x} | Energia: {energias[x]:.1f}", (30, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv.putText(warp_bgr, f"Interrupcion: {interrupciones[x]}", (30, 90), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv.imshow("Warp Actual", warp_bgr)
        cv.imshow("Referencia Warp", ref)
        cv.imshow("Diferencia Warp", diff)

    # Creamos las ventanas permitiendo que sean redimensionables
    cv.namedWindow("Warp Actual", cv.WINDOW_NORMAL)
    cv.namedWindow("Referencia Warp", cv.WINDOW_NORMAL)
    cv.namedWindow("Diferencia Warp", cv.WINDOW_NORMAL)

    cv.resizeWindow("Warp Actual", 450, 450)
    cv.resizeWindow("Referencia Warp", 450, 450)
    cv.resizeWindow("Diferencia Warp", 450, 450)

    cv.createTrackbar("Frame", "Warp Actual", 0, len(video_warps) - 1, on_trackbar)
    on_trackbar(0)
    cv.waitKey(0)
    cv.destroyAllWindows()

def ejecutar_foto_captura(vivo=True, url='', ms=250, N_estables=2, umbral=150, umbral_minimo=20, umbral_pieza=40):
    """Flujo principal procesando todo directamente en el espacio proyectado (warp)."""
    cap = cv.VideoCapture(0 if vivo else url)
    if not cap.isOpened(): return [], [], [], []

    video_warps, energias_hist, interrupciones, referencias_warps = [], [], [], []
    parser, board_logico, frame_ref_warp = None, None, None
    
    fps = 30 if vivo else cap.get(cv.CAP_PROP_FPS)
    salto = max(1, int(fps * ms / 1000.0))
    idx_frame, frames_estables = 0, 0
    post_interrupcion, pendiente_ref = False, False

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        idx_frame += 1
        if idx_frame % salto != 0: continue
        
        gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Inicialización única del parser
        if parser is None:
            if get_energia(gris) < 50: continue
            try:
                parser = inicializar_tablero(gris)
                board_logico = chess.Board()
                frame_ref_warp = parser.aplicar_warp(gris)
            except Exception as e:
                print(f"Fallo inicialización: {e}")
            continue

        # Trabajamos únicamente con la matriz alineada (warp) a partir de aquí
        gris_warp = parser.aplicar_warp(gris)
        energia = get_energia(cv.absdiff(gris_warp, frame_ref_warp))
        interrupcion = energia > umbral
        
        video_warps.append(gris_warp.copy())
        referencias_warps.append(frame_ref_warp.copy())
        energias_hist.append(energia)
        interrupciones.append(interrupcion)

        if interrupcion:
            frames_estables, post_interrupcion, pendiente_ref = 0, True, False
        else:
            frames_estables += 1

        if pendiente_ref and not interrupcion and not post_interrupcion and energia < umbral_minimo:
            frame_ref_warp, pendiente_ref, frames_estables = gris_warp.copy(), False, 0
            
        elif not interrupcion and frames_estables >= N_estables and post_interrupcion:
            if energia >= umbral_minimo:
                print(f"\n>>> DETECCION (E={energia:.1f})")
                ref_nueva_warp = gris_warp.copy()
                
                cambiadas, celdas_e = obtener_celdas_cambiadas(frame_ref_warp, ref_nueva_warp, parser, umbral_pieza)
                if len(cambiadas) >= 2:
                    inferir_movimiento_legal(board_logico, cambiadas, celdas_e)
                
                frame_ref_warp, frames_estables = ref_nueva_warp, 0
                post_interrupcion, pendiente_ref = False, True
            else:
                frame_ref_warp, frames_estables, post_interrupcion = gris_warp.copy(), 0, False

        elif not interrupcion and frames_estables >= N_estables and not post_interrupcion and energia < umbral_minimo:
            frame_ref_warp, frames_estables = gris_warp.copy(), 0

    cap.release()
    cv.destroyAllWindows()
    return video_warps, energias_hist, interrupciones, referencias_warps

if __name__ == '__main__':
    ruta = os.path.join(carpeta_data_raw, "Prueba_Completa.mp4")
    # Los umbrales bajaron porque ahora no sumamos el ruido de fuera del tablero
    video_warps, energias, interrupciones, referencias_warps = ejecutar_foto_captura(
        vivo=False, url=ruta, ms=250, N_estables=2, umbral=150, umbral_minimo=20, umbral_pieza=40
    )
    ver_por_frame(video_warps, energias, interrupciones, referencias_warps)