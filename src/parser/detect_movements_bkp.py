import cv2 as cv
import numpy as np
import time
import os
import sys
from collections import deque

# python src/parser/detect_movements.py

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

    Presionar cualquier tecla para cerrar.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def ejecutar_foto_captura(
    vivo=True,
    url='',
    ms=500,
    actualizar_ref_ms=5000,
    refresco_periodico_ms=2000,
    N_estables=5,
    umbral=1500,
    umbral_pieza=500,
    lado=800,
    energia_minima_refresco=30,
    max_intentos_inicializacion=10,  # NUEVO: límite de intentos
    timeout_segundos=30,  # NUEVO: tiempo máximo de ejecución
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
    tablero = None
    frame_ref = None
    historial_energia = deque(maxlen=10)
    
    # Contadores para debug
    intentos_inicializacion = 0
    frames_procesados = 0
    tiempo_inicio = time.time()

    # ── MODO OFFLINE ─────────────────────────────────────────────────────────
    if not vivo:
        t = 0
        ultimo_refresco_ms = 0
        ultimo_refresco_periodico_ms = 0
        frames_estables = 0
        ultimo_estado = None
        
        # Obtener duración total del video
        duracion_total_ms = int(cap.get(cv.CAP_PROP_FRAME_COUNT) * 1000 / cap.get(cv.CAP_PROP_FPS))
        print(f"Duración del video: {duracion_total_ms/1000:.1f} segundos")

        while True:
            # ── CONTROL DE TIMEOUT ──────────────────────────────────────
            tiempo_transcurrido = time.time() - tiempo_inicio
            if tiempo_transcurrido > timeout_segundos:
                print(f"⏰ TIMEOUT: {timeout_segundos}s alcanzado. Saliendo...")
                break
                
            # ── CONTROL DE AVANCE ────────────────────────────────────────
            if t > duracion_total_ms:
                print(f"✅ Video procesado completamente. Total frames: {frames_procesados}")
                break

            # ── LECTURA DE FRAME ────────────────────────────────────────
            cap.set(cv.CAP_PROP_POS_MSEC, t)
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️ No se pudo leer el frame en t={t}ms. Avanzando...")
                t += ms
                continue

            gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            energia_frame = get_energia(gris)
            frames_procesados += 1
            
            # Mostrar progreso cada 5 segundos
            if frames_procesados % 10 == 0:
                print(f"Progreso: {t/duracion_total_ms*100:.1f}% ({t}ms / {duracion_total_ms}ms)")

            # ── INICIALIZACIÓN ──────────────────────────────────────────
            if frame_ref is None:
                if energia_frame < 50:
                    print(f"[{t}ms] Frame oscuro ({energia_frame:.1f}). Avanzando...")
                    t += ms
                    continue
                
                try:
                    parser, tablero = inicializar_tablero(gris, lado)
                    frame_ref = gris.copy()
                    ultimo_refresco_ms = t
                    ultimo_refresco_periodico_ms = t
                    print(f"✅ [{t}ms] Referencia inicial establecida.")
                    intentos_inicializacion = 0
                except Exception as e:
                    intentos_inicializacion += 1
                    print(f"❌ [{t}ms] Inicialización fallida (intento {intentos_inicializacion}): {e}")
                    
                    if intentos_inicializacion >= max_intentos_inicializacion:
                        print("🔴 Demasiados intentos fallidos. Abortando...")
                        break
                    
                    t += ms
                    continue

            # ── PROCESAMIENTO NORMAL ────────────────────────────────────
            video.append(frame)
            referencias.append(frame_ref.copy())

            diff = cv.absdiff(gris, frame_ref)
            energia = get_energia(diff)
            historial_energia.append(energia)
            energia_estable = np.mean(historial_energia) if historial_energia else energia
            interrupcion = energia_estable > umbral
            
            energias.append(energia)
            interrupciones.append(interrupcion)

            # ── GESTIÓN DE ESTADOS ──────────────────────────────────────
            if interrupcion:
                frames_estables = 0
                if ultimo_estado != 'interrupcion':
                    print(f"⚡ [{t}ms] INTERRUPCIÓN (E={energia:.1f}, promedio={energia_estable:.1f})")
                    ultimo_estado = 'interrupcion'
            else:
                frames_estables += 1
                if ultimo_estado != 'estable':
                    print(f"🟢 [{t}ms] ESTABLE (E={energia:.1f}, frames={frames_estables})")
                    ultimo_estado = 'estable'

            # ── REFRESCO PERIÓDICO ──────────────────────────────────────
            tiempo_desde_ref_periodico = t - ultimo_refresco_periodico_ms
            if (not interrupcion and 
                tiempo_desde_ref_periodico >= refresco_periodico_ms and
                energia < energia_minima_refresco):
                
                print(f"🔄 [{t}ms] Refresco periódico (E={energia:.1f})")
                frame_ref = gris.copy()
                ultimo_refresco_periodico_ms = t
                ultimo_refresco_ms = t
                frames_estables = 0

            # ── DETECCIÓN DE MOVIMIENTO ─────────────────────────────────
            tiempo_desde_movimiento = t - ultimo_refresco_ms
            if (not interrupcion and 
                tiempo_desde_movimiento >= actualizar_ref_ms and 
                frames_estables >= N_estables and
                energia > energia_minima_refresco):
                
                print(f"\n🎯 [{t}ms] MOVIMIENTO DETECTADO")
                print(f"   Energía: {energia:.1f}, Frames estables: {frames_estables}")
                ref_nueva = gris.copy()

                top4, energias_celdas = obtener_top_celdas(
                    frame_ref, ref_nueva, parser, umbral_pieza
                )
                print(f"   Top celdas: {top4}")

                if len(top4) >= 2:
                    movimiento_exitoso = inferir_movimiento(tablero, top4)
                    if movimiento_exitoso:
                        print(f"   ✅ Movimiento aplicado")
                    else:
                        print(f"   ❌ No se pudo aplicar movimiento")
                else:
                    print(f"   ⚠️ Menos de 2 celdas con energía > {umbral_pieza}")

                frame_ref = ref_nueva
                ultimo_refresco_ms = t
                ultimo_refresco_periodico_ms = t
                frames_estables = 0
                ultimo_estado = None

            # ── AVANZAR AL SIGUIENTE FRAME ─────────────────────────────
            t += ms

    # ── MODO EN VIVO ─────────────────────────────────────────────────────────
    else:
        ultimo_t = 0
        ultimo_refresco = 0
        ultimo_refresco_periodico = 0
        frames_estables = 0
        ultimo_estado = None
        historial_energia = deque(maxlen=10)
        frames_sin_movimiento = 0  # NUEVO: contador de frames sin movimiento

        while cap.isOpened():
            # ── CONTROL DE TIMEOUT ──────────────────────────────────────
            tiempo_transcurrido = time.time() - tiempo_inicio
            if tiempo_transcurrido > timeout_segundos:
                print(f"⏰ TIMEOUT: {timeout_segundos}s alcanzado. Saliendo...")
                break

            ret, frame = cap.read()
            if not ret:
                print("📹 Fin del video/stream")
                break

            ahora = time.perf_counter()

            # ── MOSTRAR FRAME SIEMPRE ──────────────────────────────────
            cv.imshow("En vivo", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):  # Usar 'q' para salir
                print("🛑 Salida solicitada por el usuario")
                break

            if (ahora - ultimo_t) * 1000 < ms:
                continue

            ultimo_t = ahora
            gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            energia_frame = get_energia(gris)
            frames_procesados += 1

            # ── INICIALIZACIÓN ──────────────────────────────────────────
            if frame_ref is None:
                if energia_frame < 50:
                    continue
                
                try:
                    parser, tablero = inicializar_tablero(gris, lado)
                    frame_ref = gris.copy()
                    ultimo_refresco = ahora
                    ultimo_refresco_periodico = ahora
                    print("✅ Referencia inicial establecida (LIVE).")
                    intentos_inicializacion = 0
                except Exception as e:
                    intentos_inicializacion += 1
                    print(f"❌ Inicialización fallida (intento {intentos_inicializacion}): {e}")
                    if intentos_inicializacion >= max_intentos_inicializacion:
                        print("🔴 Demasiados intentos fallidos. Saliendo...")
                        break
                    continue

            # ── PROCESAMIENTO NORMAL ────────────────────────────────────
            video.append(frame)
            referencias.append(frame_ref.copy())

            diff = cv.absdiff(gris, frame_ref)
            energia = get_energia(diff)
            historial_energia.append(energia)
            energia_estable = np.mean(historial_energia) if historial_energia else energia
            interrupcion = energia_estable > umbral
            
            energias.append(energia)
            interrupciones.append(interrupcion)

            # ── GESTIÓN DE ESTADOS ──────────────────────────────────────
            if interrupcion:
                frames_estables = 0
                frames_sin_movimiento = 0
                if ultimo_estado != 'interrupcion':
                    print(f"⚡ INTERRUPCIÓN (E={energia:.1f})")
                    ultimo_estado = 'interrupcion'
            else:
                frames_estables += 1
                frames_sin_movimiento += 1
                if ultimo_estado != 'estable':
                    print(f"🟢 ESTABLE (E={energia:.1f}, frames={frames_estables})")
                    ultimo_estado = 'estable'

            # ── REFRESCO PERIÓDICO ──────────────────────────────────────
            tiempo_desde_ref_periodico = (ahora - ultimo_refresco_periodico) * 1000
            if (not interrupcion and 
                tiempo_desde_ref_periodico >= refresco_periodico_ms and
                energia < energia_minima_refresco):
                
                print(f"🔄 Refresco periódico (E={energia:.1f})")
                frame_ref = gris.copy()
                ultimo_refresco_periodico = ahora
                ultimo_refresco = ahora
                frames_estables = 0

            # ── DETECCIÓN DE MOVIMIENTO ─────────────────────────────────
            tiempo_desde_movimiento = (ahora - ultimo_refresco) * 1000
            if (not interrupcion and 
                tiempo_desde_movimiento >= actualizar_ref_ms and 
                frames_estables >= N_estables and
                energia > energia_minima_refresco):
                
                print(f"\n🎯 MOVIMIENTO DETECTADO (E={energia:.1f})")
                ref_nueva = gris.copy()

                top4, _ = obtener_top_celdas(frame_ref, ref_nueva, parser, umbral_pieza)
                print(f"   Top celdas: {top4}")

                if len(top4) >= 2:
                    inferir_movimiento(tablero, top4)

                frame_ref = ref_nueva
                ultimo_refresco = ahora
                ultimo_refresco_periodico = ahora
                frames_estables = 0
                ultimo_estado = None
                frames_sin_movimiento = 0

            # ── DETECTAR SI EL TABLERO ESTÁ CONGELADO ──────────────────
            if frames_sin_movimiento > 100:  # ~50 segundos sin movimiento
                print(f"⚠️ {frames_sin_movimiento} frames sin movimiento. Forzando refresco...")
                frame_ref = gris.copy()
                ultimo_refresco = ahora
                ultimo_refresco_periodico = ahora
                frames_estables = 0
                frames_sin_movimiento = 0

    cap.release()
    cv.destroyAllWindows()
    
    print(f"\n📊 Resumen:")
    print(f"   Frames procesados: {frames_procesados}")
    print(f"   Movimientos detectados: {len([e for e in interrupciones if e])}")
    print(f"   Energía promedio: {np.mean(energias) if energias else 0:.1f}")
    
    return video, energias, interrupciones, referencias


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ruta = os.path.join(carpeta_data_raw, "prueba_rotado_90.mp4")

    video, energias, interrupciones, referencias = ejecutar_foto_captura(
        vivo=False,
        url=ruta,
        ms=500,
        actualizar_ref_ms=200,
        N_estables=2,
        umbral=500,
        umbral_pieza=600,
    )

    ver_por_frame(video, energias, interrupciones, referencias)

