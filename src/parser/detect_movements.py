import cv2 as cv
import numpy as np
import chess
import time
import os
import sys

# python src/parser/detect_movements.py

dir_actual       = os.path.dirname(os.path.abspath(__file__))
dir_raiz         = os.path.dirname(os.path.dirname(dir_actual))
carpeta_data_raw = os.path.join(dir_raiz, "data", "raw")

sys.path.insert(0, dir_actual)
from parser_table import ParserTable


def get_energia(imagen):
    """Media de cuadrados de píxeles para medir movimiento."""
    return np.mean(imagen.astype(np.float32) ** 2)

def inicializar_tablero(frame_gris, lado=800):
    """Detecta el tablero en un frame y devuelve el parser configurado."""
    parser = ParserTable(frame_gris)
    parser.detect_board_corners()
    parser.correct_perspective(lado)
    parser.standardize_orientation()
    parser.detect_grid_lines()
    print("Tablero (Parser) inicializado correctamente.")
    return parser

def _warp_diferencia(diferencia_gris, H, lado):
    """Aplica la homografía H a la imagen diferencia."""
    return cv.warpPerspective(diferencia_gris, H, (lado, lado))

_TIPO_CHESS_A_VALOR = {
    chess.PAWN:   1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK:   4,
    chess.QUEEN:  6,
    chess.KING:   5,
}

def celda_a_square(fila, col):
    return chess.square(col, 7 - fila)

def celda_a_uci(fila, col):
    return "abcdefgh"[col] + str(8 - fila)

def chess_board_a_matriz(board_logico):
    """Convierte un chess.Board a la matriz 8x8 que consume LiveBoard."""
    matriz = np.zeros((8, 8), dtype=int)
    for square, pieza in board_logico.piece_map().items():
        fila  = 7 - chess.square_rank(square)
        col   = chess.square_file(square)
        valor = _TIPO_CHESS_A_VALOR[pieza.piece_type]
        matriz[fila][col] = valor if pieza.color == chess.WHITE else -valor
    return matriz

def obtener_celdas_cambiadas(frame_ref_gris, frame_nuevo_gris, parser, umbral_pieza, max_celdas=6):
    """Celdas del tablero cuya energía superó el umbral."""
    y_pos = parser.y_pos
    x_pos = parser.x_pos
    lado  = parser.tablero_hsv.shape[0]

    diferencia      = cv.absdiff(frame_nuevo_gris, frame_ref_gris)
    diferencia_warp = _warp_diferencia(diferencia, parser.H, lado)

    ENERGIAS = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            y1, y2 = y_pos[i], y_pos[i + 1]
            x1, x2 = x_pos[j], x_pos[j + 1]
            ENERGIAS[i, j] = get_energia(diferencia_warp[y1:y2, x1:x2])

    flat    = ENERGIAS.ravel()
    validos = np.where(flat > umbral_pieza)[0]
    if len(validos) == 0:
        return [], ENERGIAS

    ordenados = validos[np.argsort(flat[validos])[::-1]][:max_celdas]
    cambiadas = [(int(idx // 8), int(idx % 8)) for idx in ordenados]
    return cambiadas, ENERGIAS

def _celdas_afectadas(board_logico, mov):
    """Simula el movimiento y devuelve las celdas afectadas."""
    def _simbolos(b):
        return {sq: pieza.symbol() for sq, pieza in b.piece_map().items()}

    antes = _simbolos(board_logico)
    board_logico.push(mov)
    despues = _simbolos(board_logico)
    board_logico.pop()

    afectadas = set()
    for sq in set(antes) | set(despues):
        if antes.get(sq) != despues.get(sq):
            fila = 7 - chess.square_rank(sq)
            col  = chess.square_file(sq)
            afectadas.add((fila, col))
    return afectadas

def inferir_movimiento_legal(board_logico, cambiadas, energias_celdas):
    """Deduce el movimiento jugado utilizando chess.Board."""
    if len(cambiadas) < 2:
        print(f"  Menos de 2 celdas cambiadas ({cambiadas}); no se infiere.")
        return None

    turno  = board_logico.turn
    nombre = "blanco" if turno == chess.WHITE else "negro"
    set_cambiadas = set(cambiadas)

    origenes = []
    for (fila, col) in cambiadas:
        pieza = board_logico.piece_at(celda_a_square(fila, col))
        if pieza is not None and pieza.color == turno:
            origenes.append((fila, col))

    if not origenes:
        print(f"  Sin origenes del lado {nombre} entre {cambiadas}.")
        return None

    candidatos = []
    for (fo, co) in origenes:
        e_o = float(energias_celdas[fo, co])
        for (fd, cd) in cambiadas:
            if (fd, cd) == (fo, co):
                continue
            e_d      = float(energias_celdas[fd, cd])
            base_uci = celda_a_uci(fo, co) + celda_a_uci(fd, cd)
            
            for sufijo in ("", "q", "r", "b", "n"):
                try:
                    mov = chess.Move.from_uci(base_uci + sufijo)
                except ValueError:
                    continue
                if mov in board_logico.legal_moves:
                    explica = len(_celdas_afectadas(board_logico, mov) & set_cambiadas)
                    candidatos.append((explica, e_o + e_d, mov))
                    break

    if not candidatos:
        print(f"  Ningun movimiento legal entre {cambiadas} (turno={nombre}).")
        return None

    candidatos.sort(key=lambda x: (x[0], x[1]), reverse=True)
    mejor  = candidatos[0][2]
    unicos = {m.uci() for _, _, m in candidatos}
    if len(unicos) > 1:
        print(f"  Ambiguo {unicos} -> elijo {mejor.uci()} (explica/energia)")

    san = board_logico.san(mejor)
    board_logico.push(mejor)
    siguiente = "blanco" if board_logico.turn == chess.WHITE else "negro"
    print(f"  OK -- {mejor.uci()} ({san}) | turno ahora: {siguiente}")
    return mejor

# nos da un visor interactivo
def ver_por_frame(video, energias, interrupciones, referencias):
    """Visor interactivo frame a frame del video procesado."""
    if not video:
        print("No hay frames para mostrar.")
        return

    T_FRAME    = "Frame"
    T_REF      = "Referencia"
    T_DIFF     = "Diferencia"
    T_REF_DIFF = "Diff Referencias"

    def on_trackbar(x):
        frame     = video[x].copy()
        gris      = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_ref = referencias[x]
        diff = cv.absdiff(gris, frame_ref)

        if x > 0:
            diff_refs    = cv.absdiff(frame_ref, referencias[x - 1])
            energia_refs = get_energia(diff_refs)
        else:
            diff_refs    = np.zeros_like(frame_ref)
            energia_refs = 0.0

        energia      = energias[x]
        interrupcion = interrupciones[x]
        color_int    = (0, 0, 255) if interrupcion else (0, 255, 0)

        cv.putText(frame, f"Frame: {x}",             (20, 30),  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(frame, f"Energia: {energia:.2f}", (20, 65),  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(frame, f"Interr: {interrupcion}", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, color_int,   2)

        diff_refs_vis = diff_refs.copy()
        cv.putText(diff_refs_vis, f"E_ref: {energia_refs:.2f}", (20, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)

        cv.imshow(T_FRAME,    frame)
        cv.imshow(T_REF,      frame_ref)
        cv.imshow(T_DIFF,     diff)
        cv.imshow(T_REF_DIFF, diff_refs_vis)

    for title in [T_FRAME, T_REF, T_DIFF, T_REF_DIFF]:
        cv.namedWindow(title)

    cv.createTrackbar("Frame", T_FRAME, 0, len(video) - 1, on_trackbar)
    on_trackbar(0)
    cv.waitKey()
    cv.destroyAllWindows()



def ejecutar_foto_captura(
    vivo=True,
    url='',
    ms=250,
    N_estables=2,
    umbral=300,
    umbral_minimo=25,
    umbral_pieza=50,
    lado=800,
    max_intentos_inicializacion=10,
):
    """Procesa un video detectando y aplicando movimientos de ajedrez."""
    cap = cv.VideoCapture(0 if vivo else url)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir la fuente: {url!r}")
        return [], [], [], []

    video = []
    energias = []
    interrupciones = []
    referencias = []

    parser = None
    frame_ref = None
    board_logico = None  
    
    intentos_inicializacion = 0
    frames_procesados = 0
    tiempo_inicio = time.time()
    
    idx_frame = 0
    frames_estables = 0
    ultimo_estado = None
    post_interrupcion = False
    pendiente_ref = False
    
    fps = 30 if vivo else cap.get(cv.CAP_PROP_FPS)
    salto = max(1, int(fps * ms / 1000.0))
    
    print(f"FPS: {fps}, Salto: {salto} frames (~{salto*1000/fps:.0f}ms)")

    # mietras haya frames, procesamos
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        idx_frame += 1
        
        if idx_frame % salto != 0:
            continue
        
        gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames_procesados += 1

        # en el primer frame inicializamos el tablero
        if frame_ref is None:
            if get_energia(gris) < 50:
                continue
            try:
                parser = inicializar_tablero(gris, lado)
                board_logico = chess.Board()
                frame_ref = gris.copy()
                intentos_inicializacion = 0
                print(f"Referencia inicial establecida (frame {idx_frame}).")
            except Exception as e:
                intentos_inicializacion += 1
                print(f"Inicialización fallida (intento {intentos_inicializacion}): {e}")
                if intentos_inicializacion >= max_intentos_inicializacion:
                    print("Demasiados intentos fallidos. Abortando...")
                    break
                continue
        
        # actualizamos las listas con el frame actual
        video.append(frame)
        referencias.append(frame_ref.copy())
        
        diff = cv.absdiff(gris, frame_ref)
        energia = get_energia(diff)
        energias.append(energia)
        
        interrupcion = energia > umbral
        interrupciones.append(interrupcion)
        
        # si hay una interrupción, reseteamos el contador de frames estables
        if interrupcion:
            frames_estables = 0
            post_interrupcion = True
            pendiente_ref = False
            if ultimo_estado != 'interrupcion':
                print(f"⚡ INTERRUPCIÓN (E={energia:.1f})")
                ultimo_estado = 'interrupcion'
        else:
            frames_estables += 1
            if ultimo_estado != 'estable':
                print(f"🟢 ESTABLE (E={energia:.1f}, frames={frames_estables})")
                ultimo_estado = 'estable'
        
        if pendiente_ref and not interrupcion and not post_interrupcion:
            if energia < umbral_minimo:
                frame_ref = gris.copy()
                pendiente_ref = False
                frames_estables = 0
                ultimo_estado = None
                print(f"  🔄 [ref asentada, E={energia:.1f}]")
        
        elif (
            not interrupcion
            and frames_estables >= N_estables
            and post_interrupcion
        ):
            if energia >= umbral_minimo:
                print(f"\n>>> DETECCION (E={energia:.1f}, frames={frames_estables})")
                ref_nueva = gris.copy()
                
                cambiadas, energias_celdas = obtener_celdas_cambiadas(
                    frame_ref, ref_nueva, parser, umbral_pieza)
                print(f"  Celdas cambiadas: {cambiadas}")
                
                if len(cambiadas) >= 2:
                    mov = inferir_movimiento_legal(board_logico, cambiadas, energias_celdas)
                    if mov is not None:
                        print(f"  ✅ Movimiento aplicado: {mov.uci()}")
                    else:
                        print(f"  ❌ No se pudo aplicar movimiento")
                else:
                    print(f"  ⚠️ Menos de 2 celdas con energía > {umbral_pieza}")
                
                frame_ref = ref_nueva
                frames_estables = 0
                post_interrupcion = False
                pendiente_ref = True  
                ultimo_estado = None
            
            elif energia < umbral_minimo:
                frame_ref = gris.copy()
                frames_estables = 0
                post_interrupcion = False
                pendiente_ref = False
                ultimo_estado = None
        
        elif not interrupcion and frames_estables >= N_estables and not post_interrupcion:
            if energia < umbral_minimo:
                frame_ref = gris.copy()
                frames_estables = 0
                ultimo_estado = None
    
    cap.release()
    cv.destroyAllWindows()
    
    print(f"\n📊 Resumen:")
    print(f"   Frames leídos: {idx_frame}  |  procesados: {frames_procesados}")
    print(f"   Muestras registradas: {len(video)}")
    print(f"   Interrupciones: {sum(1 for i in interrupciones if i)}")
    print(f"   Energía promedio: {np.mean(energias) if energias else 0:.1f}")
    print(f"   Tiempo: {time.time() - tiempo_inicio:.1f}s")
    
    return video, energias, interrupciones, referencias


if __name__ == '__main__':
    ruta = os.path.join(carpeta_data_raw, "Prueba_Completa.mp4")

    video, energias, interrupciones, referencias = ejecutar_foto_captura(
        vivo=False,
        url=ruta,
        ms=250,
        N_estables=2,
        umbral=300,
        umbral_minimo=25,
        umbral_pieza=50,
    )

    ver_por_frame(video, energias, interrupciones, referencias)