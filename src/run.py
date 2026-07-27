import cv2 as cv
import chess
import numpy as np
import os
import sys

# python src/run.py

# Rutas
dir_src = os.path.dirname(os.path.abspath(__file__))
dir_raiz = os.path.dirname(dir_src)
sys.path.insert(0, os.path.join(dir_src, "parser"))
sys.path.insert(0, os.path.join(dir_src, "ui"))
sys.path.insert(0, dir_src)

from movement_detection import DetectorMovimiento, chess_board_a_matriz
from parser_table import DetectorTablero, configurar_offset
from virtual_board import LiveBoard
from stockfish_advisor import StockfishAdvisor
from menu import iniciar_menu
from viewer import visualizar_partida
from data_logger import GameLogger

# ── Configuración ─────────────────────────────────────────────────────────────
VIVO = True
URL = os.path.join(dir_raiz, "data", "raw", "Prueba2.mp4")
CAMERA_ID = 1
MAX_FRAMES_SIN_CUADRILATERO = 60

# Valores iniciales iguales a detectar_tablero.ipynb.
MOV_UMBRAL_DIFERENCIA_MEDIAS = 3
MOV_FRAMES_PAUSA = 15
MOV_MUESTRAS_STD = 10
MOV_UMBRAL_STD_X10 = 15       # 15 -> 1.5
MOV_ENERGIA_WARP_X10 = 50     # 50 -> 5.0

VENTANA_CONTROLES_MOV = "Controles movimiento"
VENTANA_ESTADOS_MOV = "Estados movimiento"
VENTANA_CHESS_VISION = "Chess Vision"


def _sin_accion(_valor):
    """Callback requerido por OpenCV; los valores se leen en cada frame."""


def _crear_controles_movimiento():
    cv.namedWindow(VENTANA_CONTROLES_MOV, cv.WINDOW_NORMAL)
    cv.createTrackbar(
        "Umbral medias",
        VENTANA_CONTROLES_MOV,
        MOV_UMBRAL_DIFERENCIA_MEDIAS,
        50,
        _sin_accion,
    )
    cv.createTrackbar(
        "Pausa frames",
        VENTANA_CONTROLES_MOV,
        MOV_FRAMES_PAUSA,
        60,
        _sin_accion,
    )
    cv.createTrackbar(
        "Muestras std",
        VENTANA_CONTROLES_MOV,
        MOV_MUESTRAS_STD,
        30,
        _sin_accion,
    )
    cv.createTrackbar(
        "Umbral std x10",
        VENTANA_CONTROLES_MOV,
        MOV_UMBRAL_STD_X10,
        200,
        _sin_accion,
    )
    cv.createTrackbar(
        "Energia warp x10",
        VENTANA_CONTROLES_MOV,
        MOV_ENERGIA_WARP_X10,
        1000,
        _sin_accion,
    )


def _actualizar_configuracion_detector(detector_mov: DetectorMovimiento):
    detector_mov.actualizar_configuracion(
        umbral_diferencia_medias=cv.getTrackbarPos(
            "Umbral medias", VENTANA_CONTROLES_MOV
        ),
        frames_pausa=cv.getTrackbarPos(
            "Pausa frames", VENTANA_CONTROLES_MOV
        ),
        cantidad_muestras_std=max(
            1,
            cv.getTrackbarPos("Muestras std", VENTANA_CONTROLES_MOV),
        ),
        umbral_std=cv.getTrackbarPos(
            "Umbral std x10", VENTANA_CONTROLES_MOV
        )
        / 10.0,
        umbral_energia_warp=cv.getTrackbarPos(
            "Energia warp x10", VENTANA_CONTROLES_MOV
        )
        / 10.0,
    )



def _extraer_warp_con_tolerancia(
    parser: DetectorTablero,
    frame: np.ndarray,
    prev_corners,
    frames_sin_cuadrilatero: int,
):
    """Obtiene el warp conservando temporalmente las últimas esquinas válidas.

    El parser ya congela esquinas ante saltos bruscos, pero lanza ValueError si
    la oclusión impide formar cualquier cuadrilátero. En ese caso seguimos
    warpeando el frame actual con las últimas esquinas hasta 60 frames, para no
    borrar el ciclo Limpia/Alerta/Interrumpida durante una jugada.
    """
    parser.update_frame(frame)

    try:
        nuevas_corners = parser.detect_board_corners(prev_corners)
    except ValueError:
        if (
            prev_corners is not None
            and frames_sin_cuadrilatero < MAX_FRAMES_SIN_CUADRILATERO
        ):
            parser.esquinas = np.asarray(prev_corners, dtype=np.float32).copy()
            try:
                warp = parser.get_board_roi()
            except (ValueError, cv.error, np.linalg.LinAlgError):
                parser.reset()
                return None, None, 0, False, True

            return (
                prev_corners,
                warp,
                frames_sin_cuadrilatero + 1,
                True,
                False,
            )

        parser.reset()
        return None, None, 0, False, True

    if nuevas_corners is None:
        parser.reset()
        return None, None, 0, False, True

    try:
        warp = parser.get_board_roi()
    except (ValueError, cv.error, np.linalg.LinAlgError):
        parser.reset()
        return None, None, 0, False, True

    usando_esquinas_congeladas = bool(
        getattr(parser, "paciencia_oclusion", 0) > 0
    )
    return nuevas_corners, warp, 0, usando_esquinas_congeladas, False


def iniciar_deteccion():
    cap = cv.VideoCapture(CAMERA_ID if VIVO else URL)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara o el archivo de video.")
        return

    if VIVO:
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)

    offset_elegido = configurar_offset(cap, offset_inicial=0)

    if not VIVO:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    parser = DetectorTablero(offset=offset_elegido)
    detector_mov = DetectorMovimiento(
        umbral_diferencia_medias=MOV_UMBRAL_DIFERENCIA_MEDIAS,
        frames_pausa=MOV_FRAMES_PAUSA,
        cantidad_muestras_std=MOV_MUESTRAS_STD,
        umbral_std=MOV_UMBRAL_STD_X10 / 10.0,
        umbral_energia_warp=MOV_ENERGIA_WARP_X10 / 10.0,
    )
    prev_corners = None
    tracking_perdido = False
    frames_sin_cuadrilatero = 0

    _crear_controles_movimiento()
    cv.namedWindow(VENTANA_ESTADOS_MOV, cv.WINDOW_NORMAL)

    live_board = LiveBoard()
    live_board.actualizar(chess_board_a_matriz(detector_mov.board_logico))
    logger = GameLogger(os.path.join(dir_raiz, "data", "log"))

    try:
        advisor = StockfishAdvisor()
    except Exception:
        advisor = None

    print(
        "\n[*] Iniciando detección de movimientos. "
        "Presiona 'q' o Esc para salir, 'r' para reiniciar la referencia visual."
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if VIVO:
                break
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            detector_mov.reiniciar_referencia_visual()
            parser.reset()
            prev_corners = None
            frames_sin_cuadrilatero = 0
            continue

        # 1. Refresco de UI con el tablero lógico encapsulado.
        if live_board is not None:
            if advisor is not None:
                live_board.sugerencias = advisor.get_sugerencias()
            if not live_board.actualizar(
                chess_board_a_matriz(detector_mov.board_logico)
            ):
                break

            if live_board.movimiento_manual:
                mov_uci = live_board.movimiento_manual
                live_board.movimiento_manual = None

                try:
                    movimiento_manual = chess.Move.from_uci(mov_uci)
                    board = detector_mov.board_logico
                    pieza_origen = board.piece_at(movimiento_manual.from_square)
                    movimiento_aplicado = False

                    if pieza_origen and pieza_origen.color == board.turn:
                        if movimiento_manual in board.legal_moves:
                            board.push(movimiento_manual)
                            movimiento_aplicado = True
                            print(f"\n[!] Movimiento manual aplicado: {mov_uci}")
                    else:
                        ultimo_movimiento = board.pop() if board.move_stack else None
                        if movimiento_manual in board.legal_moves:
                            board.push(movimiento_manual)
                            movimiento_aplicado = True
                            print(
                                f"\n[!] Corrección aplicada: {mov_uci} "
                                "(movimiento falso deshecho)"
                            )
                        else:
                            if ultimo_movimiento:
                                board.push(ultimo_movimiento)
                            print("\n[!] Movimiento manual ilegal ignorado.")

                    # La posición lógica cambió manualmente; el frame siguiente
                    # se toma como nueva referencia para evitar una detección doble.
                    if movimiento_aplicado:
                        detector_mov.reiniciar_referencia_visual()
                except (ValueError, IndexError):
                    print("\n[!] Formato de movimiento manual inválido.")

        # 2. Extracción del warp del tablero. No reiniciamos el detector ante
        # el primer frame sin cuadrilátero: podría ser la mano haciendo la jugada.
        (
            prev_corners,
            tablero_bgr,
            frames_sin_cuadrilatero,
            usando_esquinas_congeladas,
            tracking_perdido_actual,
        ) = _extraer_warp_con_tolerancia(
            parser,
            frame,
            prev_corners,
            frames_sin_cuadrilatero,
        )

        if tracking_perdido_actual:
            prev_corners = None
            frames_sin_cuadrilatero = 0
            if not tracking_perdido:
                detector_mov.reiniciar_referencia_visual()
                tracking_perdido = True
                print(
                    "\n[!] Tracking del tablero perdido definitivamente: "
                    "se reiniciaron parser y referencia de movimientos."
                )

            cv.imshow(
                VENTANA_CHESS_VISION,
                cv.resize(frame, (0, 0), fx=0.5, fy=0.5),
            )
            key = cv.waitKey(15) & 0xFF
            if key in (27, ord("q")):
                break
            continue

        tracking_perdido = False

        # 3. Configuración y detección automática usando exactamente el ciclo
        # Limpia/Alerta/Interrumpida del notebook.
        _actualizar_configuracion_detector(detector_mov)
        movimiento, san = detector_mov.procesar_roi(tablero_bgr)

        cv.imshow(
            VENTANA_ESTADOS_MOV,
            detector_mov.generar_visualizacion(tablero_bgr),
        )

        if movimiento is not None:
            turno_antes = not detector_mov.board_logico.turn
            clave_sug = "blancas" if turno_antes == chess.WHITE else "negras"
            datos_sug = live_board.sugerencias.get(clave_sug) if live_board else None
            sug_txt = datos_sug[0] if isinstance(datos_sug, tuple) else datos_sug

            logger.registrar(turno_antes, san, movimiento.uci(), sug_txt)

            if detector_mov.board_logico.is_game_over():
                live_board.set_resultado(detector_mov.board_logico.result())
                live_board.sugerencias = {"blancas": None, "negras": None}
                print("\n[*] PARTIDA FINALIZADA.")
            elif advisor is not None:
                advisor.analizar_async(detector_mov.board_logico)

        frame_mostrado = frame.copy()
        if usando_esquinas_congeladas:
            cv.putText(
                frame_mostrado,
                f"Oclusion: usando ultimas esquinas ({frames_sin_cuadrilatero}/"
                f"{MAX_FRAMES_SIN_CUADRILATERO})",
                (15, 35),
                cv.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
            )

        cv.imshow(
            VENTANA_CHESS_VISION,
            cv.resize(frame_mostrado, (0, 0), fx=0.5, fy=0.5),
        )

        # 4. Control de eventos.
        key = cv.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("r"), ord("R")):
            detector_mov.reiniciar_referencia_visual()
            print("\n[!] Referencia visual y máquinas de estado reiniciadas.")

    logger.guardar_historial_completo()
    if advisor is not None:
        advisor.cerrar()
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    iniciar_menu(iniciar_deteccion, visualizar_partida)
