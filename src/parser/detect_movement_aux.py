import cv2 as cv
import numpy as np
import chess
import time
import os
import sys

# python src/parser/detect_movement_aux.py
#
# ══════════════════════════════════════════════════════════════════════════════
# VERSIÓN AUXILIAR / DEBUG de detect_movements.py
# ══════════════════════════════════════════════════════════════════════════════
#
# ¿Por qué existe este archivo?
#   El detect_movements.py original "se freezaba" y la referencia quedaba
#   acumulada al correrlo solo (frame por frame). Acá se corrigen las TRES
#   causas, portando la lógica que YA funciona en run.py, sin tocar el original.
#
#   (1) FREEZE — el original hacía  cap.set(cv.CAP_PROP_POS_MSEC, t)  en CADA
#       iteración (seeking por ms). Con códecs H.264 en Windows el cap.read()
#       posterior se cuelga dentro de la llamada nativa. AQUÍ se lee SECUENCIAL
#       con cap.read() y se muestrea por conteo de frames (igual que run.py).
#
#   (2) REFERENCIA ACUMULADA — el original promediaba la energía con un deque
#       (media móvil de 10). Al refrescar frame_ref la diferencia caía a ~0 pero
#       el deque seguía con los valores viejos altos → la interrupción quedaba
#       "pegada" y la diferencia se actualizaba tarde. AQUÍ se usa energía
#       INSTANTÁNEA (sin deque), igual que run.py.
#
#   (3) REFERENCIA CONTAMINADA — el original refrescaba por tiempo
#       (actualizar_ref_ms) sin verificar que hubiera pasado una mano. AQUÍ se
#       usa la máquina de estados post_interrupcion / pendiente_ref de run.py:
#       solo se detecta cuando una mano entró y salió, y la referencia se vuelve
#       a limpiar recién cuando el tablero se asienta (energía < umbral_minimo).
#
# Todas las funciones de análisis son idénticas a detect_movements.py, así que
# run_aux.py puede importar de acá exactamente igual que run.py importa del
# original.

dir_actual       = os.path.dirname(os.path.abspath(__file__))
dir_raiz         = os.path.dirname(os.path.dirname(dir_actual))
carpeta_data_raw = os.path.join(dir_raiz, "data", "raw")

sys.path.insert(0, dir_actual)
from parser_table import ParserTable
from elements.board import Board


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def get_energia(imagen):
    """Media de cuadrados de píxeles.

    Proxy de cuánto cambió una zona: valor alto → mucho movimiento/diferencia.
    Se usa tanto para la señal global (¿hay interrupción?) como por celda
    (¿qué cuadros del tablero cambiaron más?).
    """
    return np.mean(imagen.astype(np.float32) ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN DEL TABLERO
# ══════════════════════════════════════════════════════════════════════════════

def inicializar_tablero(frame_gris, lado=800):
    """Detecta el tablero en un frame y construye el Board virtual.

    Ejecuta el pipeline completo de ParserTable:
      1. Esquinas del tablero (Canny + contorno convexo)
      2. Corrección de perspectiva → cuadrado lado×lado en HSV
      3. Orientación (blancas abajo, negras arriba)
      4. Grilla Hough → y_pos / x_pos (9 límites de fila y columna)

    Lanza excepción si cualquier paso falla (imagen oscura, tablero no visible,
    Hough sin líneas, etc.). El llamador decide si reintentar.

    Devuelve (parser, tablero):
      parser  : ParserTable con H, tablero_hsv, y_pos, x_pos ya calculados.
      tablero : Board en posición inicial, listo para recibir movimientos.
    """
    parser = ParserTable(frame_gris)
    parser.detect_board_corners()
    parser.correct_perspective(lado)
    parser.standardize_orientation()
    parser.detect_grid_lines()

    tablero = Board(
        nueva_partida=True,
        imagen_rectificada=parser.tablero_hsv,
        y_pos=parser.y_pos,
        x_pos=parser.x_pos,
    )
    print(f"Tablero inicializado — turno: {tablero.turno}")
    return parser, tablero


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS DE DIFERENCIA EN ESPACIO DEL TABLERO
# ══════════════════════════════════════════════════════════════════════════════

def _warp_diferencia(diferencia_gris, H, lado):
    """Aplica la homografía H a la imagen diferencia.

    Lleva la diferencia del espacio de la cámara al espacio del tablero
    rectificado, para que los recortes de cada celda sean comparables.
    H debe ser la misma matriz calculada por correct_perspective().
    """
    return cv.warpPerspective(diferencia_gris, H, (lado, lado))


def obtener_top_celdas(frame_ref_gris, frame_nuevo_gris, parser, umbral_pieza):
    """Identifica las celdas del tablero con mayor cambio entre dos frames.

    Proceso:
      1. Diferencia absoluta entre los dos frames en escala de grises.
      2. Warp de la diferencia al espacio del tablero (usa H del parser).
      3. Para cada una de las 64 celdas (según y_pos / x_pos) se calcula
         la energía del recorte de la diferencia warpeada.
      4. Se filtran las celdas cuya energía supera `umbral_pieza` y se
         devuelven las top-4 ordenadas de mayor a menor energía.

    Devuelve:
      top4     : lista de tuplas (fila, col) — máx. 4 celdas candidatas.
      ENERGIAS : np.array (8, 8) con la energía de cada celda (para debug).
    """
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

    flat            = ENERGIAS.ravel()
    indices_validos = np.where(flat > umbral_pieza)[0]

    if len(indices_validos) == 0:
        return [], ENERGIAS

    sorted_validos = indices_validos[np.argsort(flat[indices_validos])[::-1]]
    top4 = [(int(idx // 8), int(idx % 8)) for idx in sorted_validos[:4]]

    return top4, ENERGIAS


# ══════════════════════════════════════════════════════════════════════════════
# VALIDACIÓN PREVIA DE DETECCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def hay_origen_valido(tablero, top4):
    """Verifica que al menos una celda de top4 tenga pieza del turno actual.

    Si ninguna celda candidata tiene pieza del jugador en turno, la detección
    es un falso positivo (ruido, sombra, reflejo) y no debe procesarse.
    Devuelve True si hay al menos un origen posible, False si no hay ninguno.
    """
    return any(
        tablero.piezas[i][j] is not None and tablero.piezas[i][j].color == tablero.turno
        for i, j in top4
    )


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCIA Y EJECUCIÓN DEL MOVIMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def inferir_movimiento(tablero, top4):
    """Deduce origen y destino del movimiento y lo aplica al tablero virtual.

    Clasifica cada celda candidata:
      - origen  : tiene una pieza del jugador en turno.
      - destino : está vacía O tiene pieza enemiga (captura).

    Con al menos un origen y un destino, ejecuta tablero.mover().
    Si la inferencia es ambigua o el movimiento es inválido, solo imprime
    un aviso y deja el tablero sin cambios.

    Devuelve True si el movimiento se aplicó correctamente, False si no.
    """
    origenes = []
    destinos = []

    for i, j in top4:
        pieza = tablero.piezas[i][j]
        if pieza is None:
            destinos.append((i, j))
        elif pieza.color == tablero.turno:
            origenes.append((i, j))
        else:
            # pieza enemiga → celda de captura
            destinos.append((i, j))

    print(f"  turno={tablero.turno} | orígenes={origenes} | destinos={destinos}")

    if not origenes or not destinos:
        print("  No se pudo inferir movimiento (orígenes o destinos vacíos).")
        return False

    # Probar todas las combinaciones origen×destino hasta encontrar una válida
    for origen in origenes:
        for destino in destinos:
            try:
                tablero.mover(origen=origen, destino=destino)
                print(f"  OK — {origen} → {destino} | turno ahora: {tablero.turno}")
                return True
            except ValueError as e:
                print(f"  ({origen}→{destino}) inválido: {e}")

    print("  Ninguna combinación de origen×destino resultó válida.")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# LÓGICA CON python-chess  (puerto de LegalMoveDetector.calculate_move)
# ══════════════════════════════════════════════════════════════════════════════
#
# En lugar de leer la ocupación de cada celda con una CNN (como el proyecto
# original con ChessImageClassifier), usamos la ENERGÍA por celda como detector
# de "qué celdas cambiaron". Sobre ese conjunto de celdas cambiadas aplicamos la
# misma lógica de despacho por patrón + validación de legalidad, pero con
# python-chess como motor (jaque, clavadas, enroque, al paso y promoción).
#
# Convención de coordenadas (idéntica al proyecto original):
#   matriz[fila][col]:  fila 0 = arriba (negras),  fila 7 = abajo (blancas)
#   col 0 = columna 'a',  col 7 = columna 'h'
#   square = chess.square(col, 7 - fila)  →  8*(7-fila) + col
#   UCI    = 'abcdefgh'[col] + str(8 - fila)

# peon=1 caballo=2 alfil=3 torre=4 (en nuestra matriz rey=5 reina=6; ojo que en
# python-chess QUEEN=5 KING=6, por eso el remapeo explícito).
_TIPO_CHESS_A_VALOR = {
    chess.PAWN:   1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK:   4,
    chess.QUEEN:  6,
    chess.KING:   5,
}


def celda_a_square(fila, col):
    """(fila, col) de nuestra matriz → índice de casilla de python-chess."""
    return chess.square(col, 7 - fila)


def celda_a_uci(fila, col):
    """(fila, col) de nuestra matriz → string algebraico, p. ej. (6,4) → 'e2'."""
    return "abcdefgh"[col] + str(8 - fila)


def chess_board_a_matriz(board_logico):
    """Convierte un chess.Board a la matriz 8×8 que consume LiveBoard.

    Mantener el chess.Board como única fuente de verdad y derivar la matriz de
    él evita tener que sincronizar dos motores de reglas distintos (el enroque,
    el al paso y la promoción se aplican una sola vez, dentro de python-chess).
    """
    matriz = np.zeros((8, 8), dtype=int)
    for square, pieza in board_logico.piece_map().items():
        fila  = 7 - chess.square_rank(square)
        col   = chess.square_file(square)
        valor = _TIPO_CHESS_A_VALOR[pieza.piece_type]
        matriz[fila][col] = valor if pieza.color == chess.WHITE else -valor
    return matriz


def obtener_celdas_cambiadas(frame_ref_gris, frame_nuevo_gris, parser,
                             umbral_pieza, max_celdas=6):
    """Celdas del tablero cuya energía superó `umbral_pieza` entre dos frames.

    Igual que obtener_top_celdas pero pensado para la lógica con python-chess:
    devuelve TODAS las celdas que cambiaron (hasta `max_celdas`, ordenadas por
    energía descendente). El límite acota la explosión combinatoria al armar
    pares origen×destino y deja margen para el enroque (4 celdas cambian).

    Devuelve:
      cambiadas : lista de (fila, col) ordenada por energía desc.
      ENERGIAS  : np.array (8, 8) con la energía de cada celda (para debug/visor).
    """
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
    """Conjunto de celdas (fila, col) cuya pieza cambia al aplicar `mov`.

    Simula el movimiento y compara el mapa de piezas antes/después. Sirve como
    desempate: el movimiento correcto "explica" las celdas que la energía marcó
    como cambiadas. El enroque afecta 4 celdas (rey + torre), el al paso 3
    (origen + destino + peón capturado), una captura/movida normal 2.
    """
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
    """Deduce el movimiento jugado y lo aplica al chess.Board (puerto de calculate_move).

    Dado el conjunto de celdas que cambiaron (por energía), arma todos los pares
    origen->destino donde el origen tiene una pieza del lado a mover, y se queda
    con los que python-chess considera LEGALES. Cubre, sin casos especiales, el
    movimiento normal, la captura, el enroque, el al paso y la promoción
    (default: dama).

    Si hay varios movimientos legales, desempata por (1) cuántas de las celdas
    cambiadas explica cada uno y (2) energía combinada. Así el enroque (explica
    rey + torre = 4 celdas) gana sobre una movida simple de rey que dejaría
    celdas cambiadas sin justificar.

    Devuelve el chess.Move aplicado, o None si no hubo ninguno legal.
    """
    if len(cambiadas) < 2:
        print(f"  Menos de 2 celdas cambiadas ({cambiadas}); no se infiere.")
        return None

    turno  = board_logico.turn  # chess.WHITE (True) / chess.BLACK (False)
    nombre = "blanco" if turno == chess.WHITE else "negro"
    set_cambiadas = set(cambiadas)

    # Orígenes posibles: celdas cambiadas con una pieza del lado a mover.
    origenes = []
    for (fila, col) in cambiadas:
        pieza = board_logico.piece_at(celda_a_square(fila, col))
        if pieza is not None and pieza.color == turno:
            origenes.append((fila, col))

    if not origenes:
        print(f"  Sin origenes del lado {nombre} entre {cambiadas}.")
        return None

    candidatos = []  # (celdas_explicadas, energia_combinada, chess.Move)
    for (fo, co) in origenes:
        e_o = float(energias_celdas[fo, co])
        for (fd, cd) in cambiadas:
            if (fd, cd) == (fo, co):
                continue
            e_d      = float(energias_celdas[fd, cd])
            base_uci = celda_a_uci(fo, co) + celda_a_uci(fd, cd)
            # '' cubre movimiento/captura/enroque/al paso; los sufijos, la
            # promocion (se prueba dama primero, que es lo mas frecuente).
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


# ══════════════════════════════════════════════════════════════════════════════
# VISOR DE DEBUG
# ══════════════════════════════════════════════════════════════════════════════

def ver_por_frame(video, energias, interrupciones, referencias):
    """Visor interactivo frame a frame del video procesado.

    Abre 4 ventanas controladas por un slider:
      - Frame        : imagen original con energía e indicador de interrupción.
      - Referencia   : frame gris de referencia activo en ese instante.
      - Diferencia   : absdiff entre frame y su referencia.
      - Diff Refs    : absdiff entre la referencia actual y la anterior
                       (útil para ver cuándo se actualizó la referencia).

    Diferencias con el visor original:
      - El loop de espera usa cv.waitKey(30) en bucle (en vez de cv.waitKey()
        a secas). En Windows eso mantiene la GUI viva mientras se arrastra el
        slider, así no se siente "freezado". Sale con cualquier tecla o al
        cerrar la ventana.
      - Se valida que las cuatro listas tengan el mismo largo antes de indexar.
    """
    if not video:
        print("No hay frames para mostrar.")
        return

    n = min(len(video), len(energias), len(interrupciones), len(referencias))
    if not (len(video) == len(energias) == len(interrupciones) == len(referencias)):
        print(f"⚠️ Listas desalineadas (video={len(video)}, energias={len(energias)}, "
              f"interr={len(interrupciones)}, refs={len(referencias)}). Uso n={n}.")

    T_FRAME    = "Frame"
    T_REF      = "Referencia"
    T_DIFF     = "Diferencia"
    T_REF_DIFF = "Diff Referencias"

    def on_trackbar(x):
        x = max(0, min(x, n - 1))
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

    cv.createTrackbar("Frame", T_FRAME, 0, n - 1, on_trackbar)
    on_trackbar(0)

    print("Visor abierto. Arrastrá el slider 'Frame'. Cualquier tecla o cerrar la ventana para salir.")
    while True:
        k = cv.waitKey(30)
        if k != -1:                                                   # alguna tecla
            break
        if cv.getWindowProperty(T_FRAME, cv.WND_PROP_VISIBLE) < 1:    # ventana cerrada
            break
    cv.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL  (versión corregida — sin freeze, sin referencia acumulada)
# ══════════════════════════════════════════════════════════════════════════════

def ejecutar_foto_captura(
    vivo=False,
    url='',
    ms=250,
    ms_min_refresco=1000,
    N_estables=2,
    umbral=300,
    umbral_minimo=25,
    umbral_pieza=50,
    lado=800,
    max_intentos_inicializacion=10,
    mostrar_en_vivo=True,
):
    """Procesa un video detectando y aplicando movimientos de ajedrez.

    Versión auxiliar: misma estructura de retorno que el original
    (video, energias, interrupciones, referencias) para alimentar ver_por_frame,
    pero con la lógica de referencia/energía de run.py (la que SÍ anda).

    Claves de la corrección:
      - Lectura SECUENCIAL con cap.read() + muestreo por conteo de frames
        (salto = fps * ms / 1000). Sin cap.set(POS_MSEC) → sin freeze.
      - Energía INSTANTÁNEA  get_energia(absdiff(gris, frame_ref))  → sin deque,
        sin acumulación: cuando se refresca la referencia, la energía cae a ~0 ya.
      - Máquina de estados post_interrupcion / pendiente_ref: solo se detecta
        tras una mano (interrupción) y la referencia se vuelve a limpiar recién
        cuando el tablero se asienta (energia < umbral_minimo).
    """
    cap = cv.VideoCapture(0 if vivo else url)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir la fuente: {url!r}")
        return [], [], [], []

    # Muestreo por conteo de frames (en vez de seeking por ms). En vivo la cámara
    # ya entrega ~tiempo real, así que procesamos todos los frames disponibles.
    if vivo:
        salto = 1
    else:
        fps   = cap.get(cv.CAP_PROP_FPS) or 30.0
        salto = max(1, int(round(fps * ms / 1000.0)))
        print(f"FPS={fps:.1f} → muestreo cada {salto} frames (~{ms} ms).")

    video          = []
    energias       = []
    interrupciones = []
    referencias    = []

    parser            = None
    tablero           = None
    frame_ref         = None
    frames_estables   = 0
    ultimo_estado     = None
    post_interrupcion = False
    pendiente_ref     = False   # tras detección: limpiar frame_ref al asentarse
    frames_desde_ref  = 0       # cooldown post-detección (en muestras)

    min_muestras_refresco   = max(1, int(round(ms_min_refresco / max(ms, 1))))
    intentos_inicializacion = 0
    idx_frame               = 0
    frames_procesados       = 0
    tiempo_inicio           = time.time()

    while True:
        ret, frame = cap.read()           # ← SECUENCIAL: nunca se cuelga
        if not ret:
            print("📹 Fin del video/stream")
            break

        idx_frame += 1
        # Muestreo: solo procesar 1 de cada `salto` frames (sin seeking).
        if (idx_frame - 1) % salto != 0:
            continue

        if mostrar_en_vivo:
            cv.imshow("Procesando", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Salida solicitada por el usuario")
                break

        gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames_procesados += 1

        # ── INICIALIZACIÓN ──────────────────────────────────────────────────
        if frame_ref is None:
            if get_energia(gris) < 50:
                continue
            try:
                parser, tablero = inicializar_tablero(gris, lado)
                frame_ref        = gris.copy()
                frames_desde_ref = 0
                intentos_inicializacion = 0
                print("✅ Referencia inicial establecida.")
            except Exception as e:
                intentos_inicializacion += 1
                print(f"❌ Inicialización fallida (intento {intentos_inicializacion}): {e}")
                if intentos_inicializacion >= max_intentos_inicializacion:
                    print("🔴 Demasiados intentos fallidos. Abortando.")
                    break
                continue

        # ── ENERGÍA INSTANTÁNEA ─────────────────────────────────────────────
        energia      = get_energia(cv.absdiff(gris, frame_ref))
        interrupcion = energia > umbral
        frames_desde_ref += 1

        # Registro alineado para el visor (las 4 listas crecen juntas).
        video.append(frame)
        referencias.append(frame_ref.copy())
        energias.append(energia)
        interrupciones.append(interrupcion)

        # ── GESTIÓN DE ESTADOS ──────────────────────────────────────────────
        if interrupcion:
            frames_estables   = 0
            post_interrupcion = True
            pendiente_ref     = False   # nueva jugada: cancelar limpieza pendiente
            if ultimo_estado != 'interrupcion':
                print(f"⚡ INTERRUPCION  energia={energia:.1f}")
                ultimo_estado = 'interrupcion'
        else:
            frames_estables += 1
            if ultimo_estado != 'estable':
                print(f"🟢 quieto  energia={energia:.1f}  frames_estables={frames_estables}")
                ultimo_estado = 'estable'

        # ── LIMPIEZA DE REFERENCIA PENDIENTE ────────────────────────────────
        # Tras una detección el tablero puede tardar en asentarse. Apenas la
        # energía baja lo suficiente, se actualiza frame_ref (sin esperar cooldown).
        if pendiente_ref and not interrupcion and not post_interrupcion:
            if energia < umbral_minimo:
                frame_ref        = gris.copy()
                pendiente_ref    = False
                frames_estables  = 0
                frames_desde_ref = 0
                ultimo_estado    = None
                print(f"  [ref asentada  energia={energia:.1f}]")

        # ── DETECCIÓN / REFRESCO (gated por cooldown) ───────────────────────
        elif (not interrupcion
              and frames_desde_ref >= min_muestras_refresco
              and frames_estables >= N_estables):

            if post_interrupcion and energia >= umbral_minimo:
                # ── Jugada detectada ────────────────────────────────────────
                print(f"\n>>> DETECCION  energia={energia:.1f}  frames_estables={frames_estables}")
                ref_nueva = gris.copy()

                top4, energias_celdas = obtener_top_celdas(
                    frame_ref, ref_nueva, parser, umbral_pieza)
                print(f"   Top celdas: {top4}")

                if len(top4) >= 2:
                    if inferir_movimiento(tablero, top4):
                        print("   ✅ Movimiento aplicado")
                    else:
                        print("   ❌ No se pudo aplicar movimiento")
                else:
                    print(f"   ⚠️ Menos de 2 celdas con energía > {umbral_pieza}")

                frame_ref         = ref_nueva
                frames_estables   = 0
                frames_desde_ref  = 0
                post_interrupcion = False
                pendiente_ref     = True   # pedir limpieza al asentarse
                ultimo_estado     = None

            elif energia < umbral_minimo:
                # Tablero quieto sin interrupción previa (mano que no movió pieza).
                frame_ref         = gris.copy()
                frames_estables   = 0
                frames_desde_ref  = 0
                post_interrupcion = False
                pendiente_ref     = False
                ultimo_estado     = None

            else:
                # Energía elevada por deriva de cámara/luz, sin interrupción
                # previa. No se toca frame_ref; se espera a que la energía baje.
                post_interrupcion = False

    cap.release()
    cv.destroyAllWindows()

    print(f"\n📊 Resumen:")
    print(f"   Frames leídos: {idx_frame}  |  procesados: {frames_procesados}")
    print(f"   Muestras registradas: {len(video)}")
    print(f"   Interrupciones: {sum(1 for i in interrupciones if i)}")
    print(f"   Energía promedio: {np.mean(energias) if energias else 0:.1f}")
    print(f"   Tiempo: {time.time() - tiempo_inicio:.1f}s")

    return video, energias, interrupciones, referencias


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ruta = os.path.join(carpeta_data_raw, "Prueba_Completa.mp4")

    video, energias, interrupciones, referencias = ejecutar_foto_captura(
        vivo=False,
        url=ruta,
        ms=250,
        ms_min_refresco=1000,
        N_estables=2,
        umbral=300,
        umbral_minimo=25,
        umbral_pieza=50,
        mostrar_en_vivo=True,
    )

    ver_por_frame(video, energias, interrupciones, referencias)
