import cv2 as cv
import time
import os
import sys

# python src/run.py  (ejecutar desde la raíz del proyecto)

dir_src  = os.path.dirname(os.path.abspath(__file__))
dir_raiz = os.path.dirname(dir_src)

sys.path.insert(0, dir_src)

from parser.detect_movements import (
    get_energia, inicializar_tablero, obtener_top_celdas, inferir_movimiento,
)
from ui.virtual_board import LiveBoard

# ── Configuración ─────────────────────────────────────────────────────────────

VIVO            = False
URL             = os.path.join(dir_raiz, "data", "raw", "partida_larga_normal.mp4")
MS_MUESTREO     = 250    # cada cuántos ms se analiza un frame
MS_MIN_REFRESCO = 500    # tiempo mínimo entre detecciones de movimiento
N_ESTABLES      = 2      # frames quietos consecutivos para declarar reposo
UMBRAL          = 500    # energía global que indica movimiento/mano en escena
UMBRAL_MINIMO   = 30     # por debajo de esto el tablero no cambió nada real
UMBRAL_PIEZA    = 400    # energía mínima por celda para considerarla candidata
LADO            = 800    # lado del tablero rectificado en píxeles


# ── Loop principal ─────────────────────────────────────────────────────────────

def main():
    cap = cv.VideoCapture(0 if VIVO else URL)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir {'cámara' if VIVO else URL!r}")
        return

    parser          = None
    tablero         = None
    frame_ref       = None
    live_board      = None
    ultimo_t        = 0.0   # ms del último frame analizado
    ultimo_refresco = 0.0   # ms de la última actualización de referencia
    frames_estables = 0
    ultimo_estado   = None  # para imprimir solo cuando cambia el estado

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Mostrar video; 'q' para salir
        cv.imshow("Chess Vision", frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

        # Mantener pygame responsivo en cada frame (no solo al detectar movimiento)
        if live_board is not None and not live_board.actualizar(tablero.matriz):
            break

        # Controlar frecuencia de muestreo para el análisis de energía
        ahora = time.perf_counter() * 1000  # ms
        if ahora - ultimo_t < MS_MUESTREO:
            continue
        ultimo_t = ahora

        gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # ── Primera inicialización ────────────────────────────────────────────
        if frame_ref is None:
            if get_energia(gris) < 50:
                continue   # frame oscuro, esperar
            try:
                parser, tablero = inicializar_tablero(gris, LADO)
                frame_ref       = gris.copy()
                ultimo_refresco = ahora
                live_board = LiveBoard()
                live_board.actualizar(tablero.matriz)  # mostrar posición inicial
                print("Tablero inicializado. Buscar ventana 'Tablero Digital en Tiempo Real'.")
            except Exception as e:
                print(f"Inicialización fallida: {e}. Reintentando...")
                continue

        # ── Energía global ─────────────────────────────────────────────────────
        energia      = get_energia(cv.absdiff(gris, frame_ref))
        interrupcion = energia > UMBRAL

        if interrupcion:
            frames_estables = 0
            if ultimo_estado != 'interrupcion':
                print(f"[{ahora:.0f} ms] INTERRUPCION  energia={energia:.1f}")
                ultimo_estado = 'interrupcion'
        else:
            frames_estables += 1
            if ultimo_estado != 'estable':
                print(f"[{ahora:.0f} ms] quieto         energia={energia:.1f}  frames_estables={frames_estables}")
                ultimo_estado = 'estable'

        # ── Estado estable → detectar movimiento ──────────────────────────────
        if (
            not interrupcion
            and ahora - ultimo_refresco >= MS_MIN_REFRESCO
            and frames_estables >= N_ESTABLES
        ):
            if energia < UMBRAL_MINIMO:
                # El tablero no cambió nada: actualizar referencia sin buscar movimiento
                frame_ref       = gris.copy()
                ultimo_refresco = ahora
                frames_estables = 0
                ultimo_estado   = None
                continue

            print(f"\n>>> DETECCION  energia={energia:.1f}  frames_estables={frames_estables}  dt_ref={ahora-ultimo_refresco:.0f}ms")
            ref_nueva = gris.copy()

            top4, _ = obtener_top_celdas(frame_ref, ref_nueva, parser, UMBRAL_PIEZA)
            print(f"  Top celdas: {top4}")

            if len(top4) >= 2:
                inferir_movimiento(tablero, top4)

            print(tablero.matriz)
            frame_ref       = ref_nueva
            ultimo_refresco = ahora
            frames_estables = 0
            ultimo_estado   = None  # forzar reprint del proximo estado

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
