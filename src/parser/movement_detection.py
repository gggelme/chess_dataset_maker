import cv2
import numpy as np
import chess
from parser_table import DetectorTablero

# python src/parser/movement_detection.py

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

def _celdas_afectadas(board, move):
    """Calcula qué celdas físicas cambian con un movimiento (incluye enroque y al paso)."""
    afectadas = set()
    for sq in [move.from_square, move.to_square]:
        afectadas.add((7 - chess.square_rank(sq), chess.square_file(sq)))
    
    if board.is_castling(move):
        if move.to_square == chess.G1: afectadas.update([(7, 7), (7, 5)]) # Blancas corto
        elif move.to_square == chess.C1: afectadas.update([(7, 0), (7, 3)]) # Blancas largo
        elif move.to_square == chess.G8: afectadas.update([(0, 7), (0, 5)]) # Negras corto
        elif move.to_square == chess.C8: afectadas.update([(0, 0), (0, 3)]) # Negras largo
        
    if board.is_en_passant(move):
        afectadas.add((7 - chess.square_rank(move.from_square), chess.square_file(move.to_square)))
        
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

# ============================================================================
# CLASE DETECTOR DE MOVIMIENTOS
# ============================================================================

class DetectorMovimiento:
    """
    Detecta cambios en las celdas del tablero warpeado procesando la matriz 8x8.
    """
    def __init__(self, umbral_energia=12.0, frames_estabilidad=15):
        self.umbral_energia = umbral_energia
        self.frames_estabilidad = frames_estabilidad
        
        self.board_logico = chess.Board()
        self.referencia_medias = None
        self.buffer_medias = []

    def procesar_roi(self, roi_bgr):
        """
        Analiza el ROI recortado, calcula las medias y busca movimientos legales.
        Retorna: (movimiento_uci, movimiento_san) si lo detecta, si no (None, None).
        """
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        paso_h, paso_w = h // 8, w // 8
        
        # 1. Cálculo vectorizado de las medias de las 64 celdas
        medias_actuales = np.zeros((8, 8), dtype=np.float32)
        for f in range(8):
            for c in range(8):
                celda = gray[f*paso_h:(f+1)*paso_h, c*paso_w:(c+1)*paso_w]
                medias_actuales[f, c] = np.mean(celda)

        # Inicialización
        if self.referencia_medias is None:
            self.referencia_medias = medias_actuales.copy()
            return None, None

        # 2. Análisis de estabilidad (esperar a que la mano se retire)
        self.buffer_medias.append(medias_actuales)
        if len(self.buffer_medias) > self.frames_estabilidad:
            self.buffer_medias.pop(0)

        if len(self.buffer_medias) == self.frames_estabilidad:
            # Desviación estándar a lo largo del tiempo para cada celda
            std_temporal = np.std(self.buffer_medias, axis=0)
            
            # Si la varianza máxima es baja, la imagen está estática
            if np.max(std_temporal) < 2.0: 
                # 3. Comparar con el frame de referencia
                diff = np.abs(medias_actuales - self.referencia_medias)
                
                if np.max(diff) > self.umbral_energia:
                    return self._evaluar_cambios(diff, medias_actuales)

        return None, None

    def _evaluar_cambios(self, diff, medias_actuales):
        """Prepara los datos y llama al motor de ajedrez para inferir la jugada."""
        cambiadas_idx = np.argwhere(diff > self.umbral_energia)
        
        if len(cambiadas_idx) >= 2:
            cambiadas_list = [(int(f), int(c)) for f, c in cambiadas_idx]
            energias = {(f, c): diff[f, c] for f, c in cambiadas_list}
            
            mov, san = inferir_movimiento(self.board_logico, cambiadas_list, energias)
            
            if mov:
                print(f"♟️ Movimiento detectado: {san} ({mov})")
                self.referencia_medias = medias_actuales.copy()
                self.buffer_medias.clear()
                return mov, san
            
            # Reseteo en caso de falsas alarmas muy grandes (ej. cambio fuerte de luz)
            if len(cambiadas_list) > 6:
                print("⚠️ Cambio masivo detectado sin jugada legal. Reseteando referencia de luz.")
                self.referencia_medias = medias_actuales.copy()
                self.buffer_medias.clear()

        return None, None


if __name__ == "__main__":
    cap = cv2.VideoCapture(1) # Cambia a 1 si usas cámara externa

    parser = DetectorTablero(offset=80)
    detector_mov = DetectorMovimiento(umbral_energia=11.0, frames_estabilidad=15)
    prev_corners = None
    
    cv2.namedWindow("Camara Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tablero Warpeado", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camara Original", 450, 450)
    cv2.resizeWindow("Tablero Warpeado", 450, 450)


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        parser.update_frame(frame)
        
        try:
            prev_corners = parser.detect_board_corners(prev_corners)
            
            if prev_corners is None:
                cv2.imshow("Camara Original", frame)
                if cv2.waitKey(1) & 0xFF == 27: break
                continue
                
            roi = parser.get_board_roi()
            
            # --- DIBUJO ---
            cv2.polylines(frame, [parser.esquinas.astype(np.int32)], True, (0, 0, 255), 3)
            if parser.H is not None:
                H_inv = np.linalg.inv(parser.H)
                pasos = parser.get_math_grid()
                lado = parser.LADO_DESTINO
                for p in pasos:
                    pts_h = np.array([[[0, p], [lado, p]]], dtype=np.float32)
                    pts_v = np.array([[[p, 0], [p, lado]]], dtype=np.float32)
                    orig_h = cv2.perspectiveTransform(pts_h, H_inv)
                    orig_v = cv2.perspectiveTransform(pts_v, H_inv)
                    cv2.line(frame, tuple(map(int, orig_h[0][0])), tuple(map(int, orig_h[0][1])), (255, 0, 0), 2)
                    cv2.line(frame, tuple(map(int, orig_v[0][0])), tuple(map(int, orig_v[0][1])), (255, 0, 0), 2)

            # --- DETECCIÓN DE MOVIMIENTO ---
            
            mov, san = detector_mov.procesar_roi(roi)
            if mov:
                # Dibujamos en verde el recuadro si se detectó algo en este frame
                cv2.putText(roi, f"Jugada: {san}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow("Tablero Warpeado", roi)
            
        except ValueError:
            prev_corners = None
            
        cv2.imshow("Camara Original", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
