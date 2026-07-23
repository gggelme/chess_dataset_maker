import cv2 as cv
import numpy as np
import chess
from parser_table import DetectorTablero

# python src/parser/detect_movements.py

_TIPO_CHESS_A_VALOR = {
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, 
    chess.ROOK: 4, chess.KING: 5, chess.QUEEN: 6}

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

def obtener_celdas_cambiadas(diff_clean, umbral_area_celda=0.15, max_celdas=6):
    """
    Evalúa qué celdas tuvieron un cambio significativo de área.
    Recibe la imagen binarizada (diff_clean) en lugar de la imagen continua.
    """
    h, w = diff_clean.shape
    paso_y, paso_x = h // 8, w // 8
    
    energias = np.zeros((8, 8), dtype=np.float32)
    cambiadas = []
    
    for i in range(8):
        for j in range(8):
            # Recortar la celda de la imagen limpia/binarizada
            celda = diff_clean[i*paso_y:(i+1)*paso_y, j*paso_x:(j+1)*paso_x]
            
            # Calcular qué porcentaje de la celda es "blanco" (cambio detectado)
            porcentaje_cambio = np.count_nonzero(celda) / celda.size
            energias[i, j] = porcentaje_cambio
            
            if porcentaje_cambio > umbral_area_celda:
                cambiadas.append(((i, j), porcentaje_cambio))
                
    # Ordenar las celdas de mayor a menor cambio
    cambiadas.sort(key=lambda x: x[1], reverse=True)
    coords = [c[0] for c in cambiadas[:max_celdas]]
    
    return coords, energias

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


if __name__ == "__main__":
    VIVO = True
    VIDEO_PATH = "../../data/raw/Prueba2.mp4" 
    
    UMBRAL_PIEZA = 0.15        # 15% del área de una celda para registrar cambio
    DISPLAY_SIZE = (400, 400)

    cap = cv.VideoCapture(1 if VIVO else VIDEO_PATH)
    
    if VIVO:
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
    
    parser = DetectorTablero(offset=70)
    prev_corners = None
    
    cv.namedWindow("Camara Original", cv.WINDOW_NORMAL)
    cv.namedWindow("Referencia", cv.WINDOW_NORMAL)
    cv.namedWindow("Tablero Actual", cv.WINDOW_NORMAL)
    cv.namedWindow("Diferencia", cv.WINDOW_NORMAL)
    cv.resizeWindow("Camara Original", 400, 400)
    cv.resizeWindow("Referencia", 400, 400)
    cv.resizeWindow("Tablero Actual", 400, 400)
    cv.resizeWindow("Diferencia", 400, 400)

    board_logico = chess.Board()
    ultimo_mov_str = "Ninguno"
    warp_ref = None
    
    estado_txt = "Esperando jugada (ESPACIO)"
    color_txt = (255, 255, 255)

    print("Iniciando detección... Matriz inicial lista.")
    print("Presiona la barra ESPACIADORA para procesar un movimiento.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        parser.update_frame(frame)
        
        try:
            prev_corners = parser.detect_board_corners(prev_corners)
            tablero_bgr = parser.get_board_roi()
        except ValueError:
            cv.imshow("Camara Original", frame)
            if cv.waitKey(30) & 0xFF == 27: break
            continue

        tablero_gray = cv.cvtColor(tablero_bgr, cv.COLOR_BGR2GRAY)
        tablero_gray = cv.GaussianBlur(tablero_gray, (5, 5), 0)

        # Inicialización del fondo
        if warp_ref is None:
            warp_ref = tablero_gray.copy()
            continue
            
        # Calculamos la diferencia en vivo solo para visualización
        diff = cv.absdiff(tablero_gray, warp_ref)
        _, diff_thresh = cv.threshold(diff, 45, 255, cv.THRESH_BINARY)
        diff_clean = cv.morphologyEx(diff_thresh, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
        diff_masked = cv.bitwise_and(diff, diff, mask=diff_clean)

        # --- DIBUJOS ---
        if hasattr(parser, 'H') and parser.H is not None:
            H_inv = np.linalg.inv(parser.H)
            pasos = [int(i * parser.LADO_DESTINO // 8) for i in range(9)]
            lado = parser.LADO_DESTINO
            
            cv.polylines(frame, [parser.esquinas.astype(np.int32)], True, color_txt, 3)
            for p in pasos:
                orig_h = cv.perspectiveTransform(np.array([[[0, p], [lado, p]]], dtype=np.float32), H_inv)
                orig_v = cv.perspectiveTransform(np.array([[[p, 0], [p, lado]]], dtype=np.float32), H_inv)
                cv.line(frame, tuple(map(int, orig_h[0][0])), tuple(map(int, orig_h[0][1])), (255, 0, 0), 1)
                cv.line(frame, tuple(map(int, orig_v[0][0])), tuple(map(int, orig_v[0][1])), (255, 0, 0), 1)

        cv.putText(frame, estado_txt, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color_txt, 2)
        cv.putText(frame, f"Ultimo mov: {ultimo_mov_str}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        diff_visual = cv.normalize(diff_masked, None, 0, 255, cv.NORM_MINMAX)
        cv.imshow("Referencia", cv.resize(warp_ref, DISPLAY_SIZE))
        cv.imshow("Tablero Actual", cv.resize(tablero_gray, DISPLAY_SIZE))
        cv.imshow("Diferencia", cv.resize(diff_visual, DISPLAY_SIZE))
        cv.imshow("Camara Original", frame)

        # --- EVENTOS DE TECLADO ---
        # --- EVENTOS DE TECLADO ---
        key = cv.waitKey(30) & 0xFF
        if key == 27: # ESC
            break
            
        elif key == 13 or key == ord(' '): # 13 ES LA TECLA ENTER
            cambiadas, energias_celdas = obtener_celdas_cambiadas(diff_clean, UMBRAL_PIEZA)
            print(f"\n[*] ENTER PRESIONADO. Analizando celdas alteradas: {cambiadas}")
            
            mov, san = inferir_movimiento(board_logico, cambiadas, energias_celdas)
            
            if mov:
                ultimo_mov_str = f"{mov.uci()} ({san})"
                print(f"[+] Movimiento Lógico Validado: {ultimo_mov_str}")
                estado_txt = "Jugada Registrada"
                color_txt = (0, 255, 0)
            else:
                print("[-] Movimiento INVÁLIDO. No encaja con las reglas.")
                estado_txt = "Jugada Invalida"
                color_txt = (0, 0, 255)
            
            # Al apretar Enter, sí o sí tomamos una nueva FOTO DE REFERENCIA
            warp_ref = tablero_gray.copy()

    cap.release()
    cv.destroyAllWindows()


# if __name__ == "__main__":
#     VIVO = True
#     VIDEO_PATH = "../../data/raw/Prueba2.mp4" 
    
#     UMBRAL_MANO = 4.0          # % de cambio global para considerar oclusión (mano)
#     UMBRAL_MOVIMIENTO = 0.5    # % de cambio global para considerar que se movió una pieza
#     UMBRAL_PIEZA = 0.15        # % del área de una celda
#     FRAMES_ESTABLES_REQ = 4    # Frames consecutivos quietos para salir del estado de interrupción
    
#     DISPLAY_SIZE = (400, 400)

#     cap = cv.VideoCapture(1 if VIVO else VIDEO_PATH)
    
#     if VIVO:
#         cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
#         cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
    
#     parser = DetectorTablero(offset=70)
#     prev_corners = None
    
#     cv.namedWindow("Camara Original", cv.WINDOW_NORMAL)
#     cv.namedWindow("Referencia", cv.WINDOW_NORMAL)
#     cv.namedWindow("Tablero Actual", cv.WINDOW_NORMAL)
#     cv.namedWindow("Diferencia", cv.WINDOW_NORMAL)

#     cv.resizeWindow("Camara Original", 400, 400)
#     cv.resizeWindow("Referencia", 400, 400)
#     cv.resizeWindow("Tablero Actual", 400, 400)
#     cv.resizeWindow("Diferencia", 400, 400)

#     board_logico = chess.Board()
#     matriz_estado = chess_board_a_matriz(board_logico)
#     ultimo_mov_str = "Ninguno"
    
#     warp_ref = None
#     frames_estables = 0
#     post_interrupcion = False
    
#     print("Iniciando detección... Matriz inicial lista.")

#     while True:
#         ret, frame = cap.read()
#         if not ret: break

#         parser.update_frame(frame)
        
#         try:
#             prev_corners = parser.detect_board_corners(prev_corners)
#             tablero_bgr = parser.get_board_roi()
#         except ValueError:
#             cv.imshow("Camara Original", frame)
#             if cv.waitKey(30) & 0xFF == 27: break
#             continue

#         tablero_gray = cv.cvtColor(tablero_bgr, cv.COLOR_BGR2GRAY)
#         tablero_gray = cv.GaussianBlur(tablero_gray, (5, 5), 0)

#         if warp_ref is None:
#             warp_ref = tablero_gray.copy()
#             continue
            
#         # 1. Diferencia y limpieza morfológica
#         diff = cv.absdiff(tablero_gray, warp_ref)
#         _, diff_thresh = cv.threshold(diff, 45, 255, cv.THRESH_BINARY)
#         diff_clean = cv.morphologyEx(diff_thresh, cv.MORPH_OPEN, np.ones((3,3), np.uint8))
        
#         diff_masked = cv.bitwise_and(diff, diff, mask=diff_clean)

#         # 2. NUEVA MÉTRICA: Porcentaje de píxeles cambiados
#         porcentaje_cambio = (np.count_nonzero(diff_clean) / diff_clean.size) * 100.0
        
#         estado_txt = ""
#         color_txt = (0, 0, 0)

#         if porcentaje_cambio > UMBRAL_MANO:
#             frames_estables = 0
#             post_interrupcion = True
#             estado_txt = "Mano Detectada (Oclusion)"
#             color_txt = (0, 0, 255)
#         else:
#             if post_interrupcion:
#                 frames_estables += 1
#                 estado_txt = f"Estabilizando... ({frames_estables}/{FRAMES_ESTABLES_REQ})"
#                 color_txt = (0, 165, 255)
                
#                 if frames_estables >= FRAMES_ESTABLES_REQ:
#                     # Validar si hubo movimiento real o fue solo una sombra pasajera
#                     if porcentaje_cambio >= UMBRAL_MOVIMIENTO:
#                         # Pasamos 'diff_clean' o 'diff' enmascarado para evitar ruido en las celdas
#                         cambiadas, energias_celdas = obtener_celdas_cambiadas(diff_clean, UMBRAL_PIEZA)
                        
#                         print(f"\n[*] Movimiento detectado. Celdas alteradas: {cambiadas}")
#                         mov, san = inferir_movimiento(board_logico, cambiadas, energias_celdas)
                        
#                         if mov:
#                             ultimo_mov_str = f"{mov.uci()} ({san})"
#                             matriz_estado = chess_board_a_matriz(board_logico)
#                             print(f"[+] Movimiento VALIDO: {ultimo_mov_str}\n{matriz_estado}")
#                         else:
#                             print("[-] Movimiento INVALIDO o no inferido.")
                    
#                     # Forzar la actualización dura de la referencia post-movimiento
#                     warp_ref = tablero_gray.copy() 
#                     post_interrupcion = False
#                     frames_estables = 0
#             else:
#                 estado_txt = "Tablero Estable"
#                 color_txt = (0, 255, 0)
#                 # Si estamos estables y el cambio no llega a ser una pieza entera,
#                 # forzamos la absorción del ruido actualizando la referencia más rápido.
#                 if porcentaje_cambio < UMBRAL_MOVIMIENTO:
#                     warp_ref = cv.addWeighted(warp_ref, 0.90, tablero_gray, 0.10, 0)
#                     frames_estables = 0

#         # --- DIBUJAR SOBRE LA CÁMARA (Proyección de Grilla por Homografía) ---
#         if parser.H is not None:
#             H_inv = np.linalg.inv(parser.H)
#             pasos = parser.get_math_grid()
#             lado = parser.LADO_DESTINO
            
#             # Dibujar contorno exterior
#             cv.polylines(frame, [parser.esquinas.astype(np.int32)], True, color_txt, 3)
            
#             # Dibujar grilla 8x8 con perspectiva real
#             for p in pasos:
#                 orig_h = cv.perspectiveTransform(np.array([[[0, p], [lado, p]]], dtype=np.float32), H_inv)
#                 orig_v = cv.perspectiveTransform(np.array([[[p, 0], [p, lado]]], dtype=np.float32), H_inv)
#                 cv.line(frame, tuple(map(int, orig_h[0][0])), tuple(map(int, orig_h[0][1])), (255, 0, 0), 1)
#                 cv.line(frame, tuple(map(int, orig_v[0][0])), tuple(map(int, orig_v[0][1])), (255, 0, 0), 1)

#         cv.putText(frame, f"Estado: {estado_txt}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, color_txt, 2)
#         cv.putText(frame, f"Ultimo mov: {ultimo_mov_str}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
#         diff_visual = cv.normalize(diff_masked, None, 0, 255, cv.NORM_MINMAX)
#         cv.imshow("Referencia", cv.resize(warp_ref, DISPLAY_SIZE))
#         cv.imshow("Tablero Actual", cv.resize(tablero_gray, DISPLAY_SIZE))
#         cv.imshow("Diferencia", cv.resize(diff_visual, DISPLAY_SIZE))
        
#         cv.imshow("Camara Original", frame)

#         if cv.waitKey(30) & 0xFF == 27:
#             break

#     cap.release()
#     cv.destroyAllWindows()