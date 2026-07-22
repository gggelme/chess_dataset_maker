import cv2 as cv
import numpy as np
import chess
from parser_table import DetectorTablero

# python src/parser/detect_movements.py

_TIPO_CHESS_A_VALOR = {
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, 
    chess.ROOK: 4, chess.KING: 5, chess.QUEEN: 6
}

def chess_board_a_matriz(board_logico):
    matriz = np.zeros((8, 8), dtype=int)
    for square, pieza in board_logico.piece_map().items():
        fila = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        valor = _TIPO_CHESS_A_VALOR[pieza.piece_type]
        matriz[fila][col] = valor if pieza.color == chess.WHITE else -valor
    return matriz

def celda_a_square(fila, col):
    return chess.square(col, 7 - fila)

def celda_a_uci(fila, col):
    return "abcdefgh"[col] + str(8 - fila)

def obtener_celdas_cambiadas(diff_warp, umbral_pieza, max_celdas=6):
    h, w = diff_warp.shape
    paso_y, paso_x = h // 8, w // 8
    energias = np.zeros((8, 8), dtype=np.float32)
    
    for i in range(8):
        for j in range(8):
            celda = diff_warp[i*paso_y:(i+1)*paso_y, j*paso_x:(j+1)*paso_x]
            energias[i, j] = np.mean(celda.astype(np.float32)**2)
            
    validos = np.where(energias > umbral_pieza)
    coords = list(zip(validos[0], validos[1]))
    return sorted(coords, key=lambda c: energias[c[0], c[1]], reverse=True)[:max_celdas], energias

def _celdas_afectadas(board_logico, mov):
    antes = {sq: pieza.symbol() for sq, pieza in board_logico.piece_map().items()}
    board_logico.push(mov)
    despues = {sq: pieza.symbol() for sq, pieza in board_logico.piece_map().items()}
    board_logico.pop()

    afectadas = set()
    for sq in set(antes) | set(despues):
        if antes.get(sq) != despues.get(sq):
            afectadas.add((7 - chess.square_rank(sq), chess.square_file(sq)))
    return afectadas

def inferir_movimiento(board_logico, cambiadas, energias_celdas):
    if len(cambiadas) < 2: return None, None
    
    turno = board_logico.turn
    set_cambiadas = set(cambiadas)
    origenes = [(f, c) for f, c in cambiadas if board_logico.piece_at(celda_a_square(f, c)) and board_logico.piece_at(celda_a_square(f, c)).color == turno]
    if not origenes: return None, None

    candidatos = []
    for (fo, co) in origenes:
        for (fd, cd) in cambiadas:
            if (fd, cd) == (fo, co): continue
            base_uci = celda_a_uci(fo, co) + celda_a_uci(fd, cd)
            
            for sufijo in ("", "q", "r", "b", "n"):
                try: mov = chess.Move.from_uci(base_uci + sufijo)
                except ValueError: continue
                
                if mov in board_logico.legal_moves:
                    explica = len(_celdas_afectadas(board_logico, mov) & set_cambiadas)
                    e_total = float(energias_celdas[fo, co]) + float(energias_celdas[fd, cd])
                    candidatos.append((explica, e_total, mov))
                    break

    if not candidatos: return None, None

    candidatos.sort(key=lambda x: (x[0], x[1]), reverse=True)
    mejor_mov = candidatos[0][2]
    san = board_logico.san(mejor_mov)
    
    board_logico.push(mejor_mov)
    return mejor_mov, san

# ==========================================
# UTILIDAD VISUAL
# ==========================================
def dibujar_grilla_en_camara(frame, esquinas):
    """Interpola los puntos de las esquinas para dibujar la grilla 8x8 con perspectiva."""
    TL, TR, BR, BL = esquinas
    for i in range(1, 8):
        f = i / 8.0
        # Lineas horizontales
        izq = TL * (1 - f) + BL * f
        der = TR * (1 - f) + BR * f
        cv.line(frame, tuple(izq.astype(int)), tuple(der.astype(int)), (0, 150, 0), 1)
        # Lineas verticales
        sup = TL * (1 - f) + TR * f
        inf = BL * (1 - f) + BR * f
        cv.line(frame, tuple(sup.astype(int)), tuple(inf.astype(int)), (0, 150, 0), 1)

# ==========================================
# BUCLE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    VIVO = True
    VIDEO_PATH = "data/raw/Prueba2.mp4" 
    
    UMBRAL_INTERRUPCION = 600
    UMBRAL_PIEZA = 80
    UMBRAL_MINIMO = 30
    FRAMES_ESTABLES_REQ = 4
    DISPLAY_SIZE = (400, 400)

    cap = cv.VideoCapture(1 if VIVO else VIDEO_PATH)
    detector = DetectorTablero(vivo=VIVO, refresco_frames=500, offset=70)  
    
    board_logico = chess.Board()
    matriz_estado = chess_board_a_matriz(board_logico)
    ultimo_mov_str = "Ninguno"
    
    warp_ref = None
    frames_estables = 0
    post_interrupcion = False
    
    print("Iniciando detección... Matriz inicial lista.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        tablero_bgr, esquinas = detector.procesar_frame(frame)

        if tablero_bgr is not None:
            tablero_gray = cv.cvtColor(tablero_bgr, cv.COLOR_BGR2GRAY)
            tablero_gray = cv.GaussianBlur(tablero_gray, (5, 5), 0)

            if detector.estado == "FIJADO" and detector.contador_frames <= 1:
                warp_ref = tablero_gray.copy()
                post_interrupcion = False
                frames_estables = 0
                continue

            if warp_ref is None:
                warp_ref = tablero_gray.copy()
                continue
                
            diff = cv.absdiff(tablero_gray, warp_ref)
            energia_global = np.mean(diff.astype(np.float32) ** 2)

            if energia_global > UMBRAL_INTERRUPCION:
                frames_estables = 0
                post_interrupcion = True
                estado_txt = "INTERRUPCION"
                color_txt = (0, 0, 255)
            else:
                if post_interrupcion:
                    frames_estables += 1
                    estado_txt = f"ESTABILIZANDO ({frames_estables}/{FRAMES_ESTABLES_REQ})"
                    color_txt = (0, 165, 255)
                    
                    if frames_estables >= FRAMES_ESTABLES_REQ:
                        if energia_global >= UMBRAL_MINIMO:
                            cambiadas, energias_celdas = obtener_celdas_cambiadas(diff, UMBRAL_PIEZA)
                            
                            # DEBUG: Imprimir las celdas alteradas antes de inferir
                            print(f"[*] Celdas con energía alta: {cambiadas}")
                            
                            mov, san = inferir_movimiento(board_logico, cambiadas, energias_celdas)
                            
                            if mov:
                                ultimo_mov_str = f"{mov.uci()} ({san})"
                                matriz_estado = chess_board_a_matriz(board_logico)
                                print(f"\n[+] Movimiento: {ultimo_mov_str}\n{matriz_estado}")
                            else:
                                print("[-] No se pudo inferir un movimiento legal con esas celdas.")
                        
                        warp_ref = tablero_gray.copy() 
                        post_interrupcion = False
                        frames_estables = 0
                else:
                    estado_txt = "ESTABLE"
                    color_txt = (0, 255, 0)
                    if energia_global < (UMBRAL_MINIMO * 0.7):
                        warp_ref = cv.addWeighted(warp_ref, 0.95, tablero_gray, 0.05, 0)
                        frames_estables = 0

            # --- DIBUJOS EN LA CÁMARA ---
            dibujar_grilla_en_camara(frame, esquinas)
            cv.polylines(frame, [esquinas.astype(np.int32)], True, color_txt, 2)
            cv.putText(frame, f"Estado: {estado_txt}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color_txt, 2)
            cv.putText(frame, f"Ultimo mov: {ultimo_mov_str}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            diff_visual = cv.normalize(diff, None, 0, 255, cv.NORM_MINMAX)
            cv.imshow("Referencia", cv.resize(warp_ref, DISPLAY_SIZE))
            cv.imshow("Tablero Actual", cv.resize(tablero_gray, DISPLAY_SIZE))
            cv.imshow("Diferencia", cv.resize(diff_visual, DISPLAY_SIZE))
            
        h_cam, w_cam = frame.shape[:2]
        cv.imshow("Camara Original", cv.resize(frame, (w_cam // 2, h_cam // 2)))

        if cv.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()