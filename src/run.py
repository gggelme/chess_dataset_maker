import cv2 as cv
import chess
import time
import os
import sys

# python src/run.py  (ejecutar desde la raíz del proyecto)

dir_src  = os.path.dirname(os.path.abspath(__file__))
dir_raiz = os.path.dirname(dir_src)

sys.path.insert(0, dir_raiz)

from src.parser.detect_movements import (get_energia, inicializar_tablero, obtener_celdas_cambiadas, inferir_movimiento_legal, chess_board_a_matriz,)

from ui.virtual_board import LiveBoard
from ui.stockfish_advisor import StockfishAdvisor
from data_logger import GameLogger

# ── Configuración ─────────────────────────────────────────────────────────────

VIVO            = False
URL             = os.path.join(dir_raiz, "data", "raw", "partida_larga_normal.mp4")
MS_MUESTREO     = 250    # cada cuántos ms se analiza un frame
MS_MIN_REFRESCO = 1000   # cooldown post-detección (ms)
N_ESTABLES      = 2      # frames quietos consecutivos para declarar reposo
UMBRAL          = 300    # energía que indica movimiento/mano en escena
UMBRAL_MINIMO   = 25     # por debajo de esto el tablero no cambió nada real (ruido ~8-10)
UMBRAL_PIEZA    = 50    # energía mínima por celda para considerarla candidata
LADO            = 800    # lado del tablero rectificado en píxeles


# ── Visualización de energía ──────────────────────────────────────────────────

def _dibujar_energia(diff_warp, energias, y_pos, x_pos, umbral_pieza):
    vis   = cv.cvtColor(cv.convertScaleAbs(diff_warp, alpha=3), cv.COLOR_GRAY2BGR)
    max_e = float(max(energias.max(), 1))
    for i in range(8):
        for j in range(8):
            y1, y2 = y_pos[i], y_pos[i + 1]
            x1, x2 = x_pos[j], x_pos[j + 1]
            e      = float(energias[i, j])
            ratio  = min(e / max_e, 1.0)
            color  = (0, int(255 * (1 - ratio)), int(255 * ratio))
            grosor = 2 if e > umbral_pieza else 1
            cv.rectangle(vis, (x1, y1), (x2, y2), color, grosor)
            cv.putText(vis, str(int(e)), (x1 + 3, y1 + 16),
                       cv.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return vis


# ── Loop principal ─────────────────────────────────────────────────────────────

def main():
    cap = cv.VideoCapture(0 if VIVO else URL)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir {'cámara' if VIVO else URL!r}")
        return

    parser            = None
    board_logico      = None   
    frame_ref         = None
    live_board        = None
    advisor           = None   
    logger            = None   
    ultimo_t          = 0.0
    ultimo_refresco   = 0.0
    frames_estables   = 0
    ultimo_estado     = None
    post_interrupcion = False
    pendiente_ref     = False  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv.imshow("Chess Vision", frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

        if live_board is not None:
            if advisor is not None:
                live_board.sugerencias = advisor.get_sugerencias()
            if not live_board.actualizar(chess_board_a_matriz(board_logico)):
                break

        ahora = time.perf_counter() * 1000 
        if ahora - ultimo_t < MS_MUESTREO:
            continue
        ultimo_t = ahora

        gris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # ── Inicialización ────────────────────────────────────────────────────
        if frame_ref is None:
            if get_energia(gris) < 50:
                continue
            try:
                parser = inicializar_tablero(gris, LADO)
                board_logico    = chess.Board()
                frame_ref       = gris.copy()
                ultimo_refresco = ahora
                live_board = LiveBoard()
                live_board.actualizar(chess_board_a_matriz(board_logico))
                try:
                    advisor = StockfishAdvisor()
                    print("Stockfish listo.")
                except Exception as e:
                    print(f"Stockfish no disponible: {e}")
                logger = GameLogger(os.path.join(dir_raiz, "data", "log"))
                print("Tablero inicializado.")
            except Exception as e:
                print(f"Inicialización fallida: {e}. Reintentando...")
                continue

        # ── Energía global ────────────────────────────────────────────────────
        energia      = get_energia(cv.absdiff(gris, frame_ref))
        interrupcion = energia > UMBRAL

        if interrupcion:
            frames_estables   = 0
            post_interrupcion = True
            pendiente_ref     = False
            if ultimo_estado != 'interrupcion':
                print(f"[{ahora:.0f}ms] INTERRUPCION  energia={energia:.1f}")
                ultimo_estado = 'interrupcion'
        else:
            frames_estables += 1
            if ultimo_estado != 'estable':
                print(f"[{ahora:.0f}ms] quieto  energia={energia:.1f}  "
                      f"frames_estables={frames_estables}  "
                      f"dt_ref={ahora-ultimo_refresco:.0f}ms")
                ultimo_estado = 'estable'

        # ── Limpieza de referencia pendiente ────────────────────────────────── 
        if pendiente_ref and not interrupcion and not post_interrupcion:
            if energia < UMBRAL_MINIMO:
                frame_ref       = gris.copy()
                pendiente_ref   = False
                frames_estables = 0
                ultimo_estado   = None
                print(f"  [ref asentada  energia={energia:.1f}]")

        # ── Detección / Refresco ─────────────────────────────────────────────
        elif (not interrupcion
            and ahora - ultimo_refresco >= MS_MIN_REFRESCO
            and frames_estables >= N_ESTABLES):
            if post_interrupcion and energia >= UMBRAL_MINIMO:
                # ── Jugada detectada ─────────────────────────────────────────
                print(f"\n>>> DETECCION  energia={energia:.1f}  "
                      f"frames_estables={frames_estables}  dt_ref={ahora-ultimo_refresco:.0f}ms")
                ref_nueva = gris.copy()

                cambiadas, energias_celdas = obtener_celdas_cambiadas(
                    frame_ref, ref_nueva, parser, UMBRAL_PIEZA)
                turno = "blanco" if board_logico.turn == chess.WHITE else "negro"
                print(f"  Celdas cambiadas: {cambiadas}  turno={turno}")

                diff_warp = cv.warpPerspective(
                    cv.absdiff(frame_ref, ref_nueva), parser.H, (LADO, LADO))
                cv.imshow("Energia celdas",
                          _dibujar_energia(diff_warp, energias_celdas,
                                           parser.y_pos, parser.x_pos, UMBRAL_PIEZA))

                # Capturar estado PRE-jugada para el log y la SAN
                board_antes  = board_logico.copy()
                turno_antes  = board_logico.turn
                clave_sug    = 'blancas' if turno_antes == chess.WHITE else 'negras'
                sug_anterior = live_board.sugerencias.get(clave_sug) if live_board else None

                mov = inferir_movimiento_legal(board_logico, cambiadas, energias_celdas)

                if mov is not None and logger is not None:
                    san = board_antes.san(mov)
                    logger.registrar(turno_antes, san, mov.uci(), sug_anterior)

                if mov is not None and advisor is not None:
                    advisor.analizar_async(board_logico)

                if mov is not None:
                    print(chess_board_a_matriz(board_logico))
                    live_board.actualizar(chess_board_a_matriz(board_logico))

                frame_ref         = ref_nueva
                ultimo_refresco   = ahora
                frames_estables   = 0
                post_interrupcion = False
                pendiente_ref     = True
                ultimo_estado     = None

            elif energia < UMBRAL_MINIMO:
                frame_ref         = gris.copy()
                frames_estables   = 0
                post_interrupcion = False
                pendiente_ref     = False
                ultimo_estado     = None
            else:
                post_interrupcion = False

    if logger is not None:
        logger.guardar_historial_completo()
    if advisor is not None:
        advisor.cerrar()

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()