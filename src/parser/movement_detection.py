"""Detección de movimientos sobre el warp del tablero.

La máquina de estados replica la lógica de ``detectar_tablero.ipynb``:

* cada una de las 64 celdas mantiene un estado Limpia/Alerta/Interrumpida;
* el warp de referencia solo se actualiza cuando las 64 celdas están limpias;
* cualquier Alerta o Interrumpida congela esa referencia;
* un ciclo que solo tuvo Alertas se considera una falsa alarma;
* si el ciclo tuvo al menos una Interrumpida, al volver todas a Limpia se
  compara el warp actual con la referencia congelada;
* si la energía global supera el umbral, las dos celdas de mayor energía se
  usan para inferir una jugada legal con python-chess.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import chess
import cv2
import numpy as np

# python src/parser/movement_detection.py

_TIPO_CHESS_A_VALOR = {
    chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
    chess.ROOK: 4, chess.KING: 5, chess.QUEEN: 6,}

CELDA_LIMPIA = 0
CELDA_ALERTA = 1
CELDA_INTERRUMPIDA = 2

NOMBRES_ESTADO_CELDA = {
    CELDA_LIMPIA: "Limpia",
    CELDA_ALERTA: "Alerta",
    CELDA_INTERRUMPIDA: "Interrumpida",
}

COLORES_ESTADO_CELDA = {
    CELDA_LIMPIA: (0, 255, 0),
    CELDA_ALERTA: (0, 255, 255),
    CELDA_INTERRUMPIDA: (0, 0, 255),
}

CELDAS_POR_LADO = 8
NUM_CELDAS_TABLERO = CELDAS_POR_LADO * CELDAS_POR_LADO
TAM_CELDA_WARP = 100
TAM_WARP = CELDAS_POR_LADO * TAM_CELDA_WARP

Celda = Tuple[int, int]

def chess_board_a_matriz(board_logico: chess.Board) -> np.ndarray:
    matriz = np.zeros((8, 8), dtype=int)
    for square, pieza in board_logico.piece_map().items():
        fila = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        valor = _TIPO_CHESS_A_VALOR[pieza.piece_type]
        matriz[fila][col] = valor if pieza.color == chess.WHITE else -valor
    return matriz


def celda_a_square(fila: int, col: int) -> chess.Square:
    return chess.square(col, 7 - fila)

def celda_a_uci(fila: int, col: int) -> str:
    return "abcdefgh"[col] + str(8 - fila)

def indice_a_celda(indice: int) -> Celda:
    return divmod(int(indice), CELDAS_POR_LADO)

def indice_a_uci(indice: int) -> str:
    fila, columna = indice_a_celda(indice)
    return celda_a_uci(fila, columna)

def _celdas_afectadas(board: chess.Board, move: chess.Move) -> set[Celda]:
    """Calcula las celdas físicas que cambia una jugada legal."""
    afectadas: set[Celda] = set()
    for sq in (move.from_square, move.to_square):
        afectadas.add((7 - chess.square_rank(sq), chess.square_file(sq)))

    if board.is_castling(move):
        if move.to_square == chess.G1:
            afectadas.update([(7, 7), (7, 5)])
        elif move.to_square == chess.C1:
            afectadas.update([(7, 0), (7, 3)])
        elif move.to_square == chess.G8:
            afectadas.update([(0, 7), (0, 5)])
        elif move.to_square == chess.C8:
            afectadas.update([(0, 0), (0, 3)])

    if board.is_en_passant(move):
        afectadas.add((7 - chess.square_rank(move.from_square), chess.square_file(move.to_square),))

    return afectadas


def inferir_movimiento(board_logico: chess.Board, cambiadas: Sequence[Celda], energias_celdas: Dict[Celda, float],) -> Tuple[Optional[chess.Move], Optional[str]]:
    """Infiere una jugada legal a partir de las celdas visualmente visitadas."""
    if len(cambiadas) < 2:
        return None, None

    turno = board_logico.turn
    set_cambiadas = set(cambiadas)
    origenes = [
        (fila, columna)
        for fila, columna in cambiadas
        if (pieza := board_logico.piece_at(celda_a_square(fila, columna))) and pieza.color == turno]

    if not origenes:
        return None, None

    candidatos: List[Tuple[int, float, chess.Move]] = []
    for fila_origen, col_origen in origenes:
        for fila_destino, col_destino in cambiadas:
            if (fila_destino, col_destino) == (fila_origen, col_origen):
                continue

            base_uci = (celda_a_uci(fila_origen, col_origen) + celda_a_uci(fila_destino, col_destino))

            # Se prueban las promociones además de la jugada normal.
            for sufijo in ("", "q", "r", "b", "n"):
                try:
                    movimiento = chess.Move.from_uci(base_uci + sufijo)
                except ValueError:
                    continue

                if movimiento in board_logico.legal_moves:
                    explicadas = len(_celdas_afectadas(board_logico, movimiento) & set_cambiadas)
                    energia_total = float(energias_celdas[(fila_origen, col_origen)]) + float(energias_celdas[(fila_destino, col_destino)])
                    candidatos.append((explicadas, energia_total, movimiento))
                    break

    if not candidatos:
        return None, None

    candidatos.sort(key=lambda candidato: (candidato[0], candidato[1]), reverse=True)
    mejor_movimiento = candidatos[0][2]
    san = board_logico.san(mejor_movimiento)
    board_logico.push(mejor_movimiento)
    return mejor_movimiento, san

class DetectorMovimiento:
    """Detector basado en las 64 máquinas de estado del notebook."""

    def __init__(self, umbral_diferencia_medias: float = 3.0, frames_pausa: int = 15,
        cantidad_muestras_std: int = 10, umbral_std: float = 1.5,
        umbral_energia_warp: float = 5.0, board_logico: Optional[chess.Board] = None,) -> None:
        self.board_logico = board_logico if board_logico is not None else chess.Board()

        self.umbral_diferencia_medias = 0.0
        self.frames_pausa = 0
        self.cantidad_muestras_std = 1
        self.umbral_std = 0.0
        self.umbral_energia_warp = 0.0
        self.actualizar_configuracion(
            umbral_diferencia_medias=umbral_diferencia_medias,
            frames_pausa=frames_pausa,
            cantidad_muestras_std=cantidad_muestras_std,
            umbral_std=umbral_std,
            umbral_energia_warp=umbral_energia_warp,)

        self.estados_celdas = np.full(NUM_CELDAS_TABLERO, CELDA_LIMPIA, dtype=np.uint8)
        self.medias_frame_anterior = np.full(NUM_CELDAS_TABLERO, np.nan, dtype=np.float32)
        self.frames_pausa_por_celda = np.zeros(NUM_CELDAS_TABLERO, dtype=np.int32)
        self.muestras_medias_por_celda: List[List[float]] = [[] for _ in range(NUM_CELDAS_TABLERO)]
        self.ultima_std_por_celda = np.zeros(NUM_CELDAS_TABLERO, dtype=np.float32)
        self.ultima_diferencia_media_por_celda = np.zeros(NUM_CELDAS_TABLERO, dtype=np.float32)
        self.ultima_energia_por_celda = np.zeros(NUM_CELDAS_TABLERO, dtype=np.float32)
        self.ultima_energia_warp = 0.0

        self.warp_referencia_limpio: Optional[np.ndarray] = None
        self.ciclo_tuvo_alerta = False
        self.ciclo_tuvo_interrumpida = False
        self.inicializado = False

        self.ultimas_celdas_visitadas: List[Celda] = []
        self.ultimas_casillas_visitadas: List[str] = []
        self.ultimo_mensaje = ""

    def actualizar_configuracion(
        self,
        *,
        umbral_diferencia_medias: Optional[float] = None,
        frames_pausa: Optional[int] = None,
        cantidad_muestras_std: Optional[int] = None,
        umbral_std: Optional[float] = None,
        umbral_energia_warp: Optional[float] = None,
    ) -> None:
        """Actualiza los umbrales sin reiniciar el estado del detector."""
        if umbral_diferencia_medias is not None:
            self.umbral_diferencia_medias = max(
                0.0, float(umbral_diferencia_medias)
            )
        if frames_pausa is not None:
            self.frames_pausa = max(0, int(frames_pausa))
        if cantidad_muestras_std is not None:
            self.cantidad_muestras_std = max(1, int(cantidad_muestras_std))
        if umbral_std is not None:
            self.umbral_std = max(0.0, float(umbral_std))
        if umbral_energia_warp is not None:
            self.umbral_energia_warp = max(0.0, float(umbral_energia_warp))

    @staticmethod
    def _normalizar_warp_gris(roi: np.ndarray) -> Optional[np.ndarray]:
        """Convierte un warp válido a gris uint8 de 800x800."""
        if roi is None or not isinstance(roi, np.ndarray) or roi.size == 0:
            return None

        if roi.ndim == 3:
            if roi.shape[2] == 4:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        elif roi.ndim == 2:
            gray = roi.copy()
        else:
            return None

        if gray.shape[:2] != (TAM_WARP, TAM_WARP):
            gray = cv2.resize(
                gray, (TAM_WARP, TAM_WARP), interpolation=cv2.INTER_LINEAR
            )

        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        return gray

    @staticmethod
    def _extraer_celdas_warp_gris(
        warp_gray: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Devuelve las 64 ROIs 100x100 y sus medias, en orden row-major."""
        if warp_gray is None or warp_gray.size == 0:
            return None, None

        if warp_gray.shape[:2] != (TAM_WARP, TAM_WARP):
            warp_gray = cv2.resize(
                warp_gray, (TAM_WARP, TAM_WARP), interpolation=cv2.INTER_LINEAR
            )

        rois = np.empty(
            (NUM_CELDAS_TABLERO, TAM_CELDA_WARP, TAM_CELDA_WARP),
            dtype=np.uint8,
        )
        medias = np.empty(NUM_CELDAS_TABLERO, dtype=np.float32)

        for fila in range(CELDAS_POR_LADO):
            for columna in range(CELDAS_POR_LADO):
                indice = fila * CELDAS_POR_LADO + columna
                y1 = fila * TAM_CELDA_WARP
                x1 = columna * TAM_CELDA_WARP
                roi = warp_gray[
                    y1 : y1 + TAM_CELDA_WARP,
                    x1 : x1 + TAM_CELDA_WARP,
                ]
                rois[indice] = roi
                medias[indice] = float(np.mean(roi))

        return rois, medias

    @staticmethod
    def _calcular_energia_diferencia(
        imagen_actual: Optional[np.ndarray],
        imagen_referencia: Optional[np.ndarray],
    ) -> float:
        """Media absoluta de la diferencia entre dos imágenes grises."""
        if imagen_actual is None or imagen_referencia is None:
            return 0.0

        if imagen_actual.shape != imagen_referencia.shape:
            imagen_actual = cv2.resize(
                imagen_actual,
                (imagen_referencia.shape[1], imagen_referencia.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        return float(np.mean(cv2.absdiff(imagen_actual, imagen_referencia)))

    def reiniciar_referencia_visual(self) -> None:
        """Reinicia estados y referencia visual, conservando el tablero lógico."""
        self.estados_celdas.fill(CELDA_LIMPIA)
        self.medias_frame_anterior.fill(np.nan)
        self.frames_pausa_por_celda.fill(0)
        self.ultima_std_por_celda.fill(0.0)
        self.ultima_diferencia_media_por_celda.fill(0.0)
        self.ultima_energia_por_celda.fill(0.0)
        for muestras in self.muestras_medias_por_celda:
            muestras.clear()

        self.warp_referencia_limpio = None
        self.ciclo_tuvo_alerta = False
        self.ciclo_tuvo_interrumpida = False
        self.ultima_energia_warp = 0.0
        self.ultimas_celdas_visitadas.clear()
        self.ultimas_casillas_visitadas.clear()
        self.ultimo_mensaje = ""
        self.inicializado = False

    def reiniciar_partida(self) -> None:
        """Reinicia tanto la referencia visual como el tablero de ajedrez."""
        self.board_logico.reset()
        self.reiniciar_referencia_visual()

    def _inicializar(self, warp_gray: np.ndarray) -> bool:
        self.reiniciar_referencia_visual()
        _, medias = self._extraer_celdas_warp_gris(warp_gray)
        if medias is None:
            return False

        self.medias_frame_anterior[:] = medias
        self.warp_referencia_limpio = warp_gray.copy()
        self.inicializado = True
        return True

    def _analizar_movimiento_entre_warps(
        self, warp_actual: np.ndarray
    ) -> Tuple[Optional[chess.Move], Optional[str]]:
        """Aplica el umbral global y selecciona las dos celdas más cambiadas."""
        referencia = self.warp_referencia_limpio
        if referencia is None:
            return None, None

        self.ultima_energia_warp = self._calcular_energia_diferencia(
            warp_actual, referencia
        )

        if self.ultima_energia_warp <= self.umbral_energia_warp:
            self.ultima_energia_por_celda.fill(0.0)
            self.ultimas_celdas_visitadas.clear()
            self.ultimas_casillas_visitadas.clear()
            self.ultimo_mensaje = (
                "Ciclo descartado: energía global "
                f"{self.ultima_energia_warp:.3f} <= "
                f"{self.umbral_energia_warp:.3f}"
            )
            return None, None

        rois_actuales, _ = self._extraer_celdas_warp_gris(warp_actual)
        rois_referencia, _ = self._extraer_celdas_warp_gris(referencia)
        if rois_actuales is None or rois_referencia is None:
            return None, None

        for indice in range(NUM_CELDAS_TABLERO):
            self.ultima_energia_por_celda[indice] = (
                self._calcular_energia_diferencia(
                    rois_actuales[indice], rois_referencia[indice]
                )
            )

        indices_ordenados = np.argsort(
            -self.ultima_energia_por_celda, kind="stable"
        )
        indices_visitados = [
            int(indices_ordenados[0]),
            int(indices_ordenados[1]),
        ]
        celdas_visitadas = [indice_a_celda(indice) for indice in indices_visitados]
        energias = {
            celda: float(self.ultima_energia_por_celda[indice])
            for celda, indice in zip(celdas_visitadas, indices_visitados)
        }

        self.ultimas_celdas_visitadas = celdas_visitadas
        self.ultimas_casillas_visitadas = [
            celda_a_uci(fila, columna)
            for fila, columna in celdas_visitadas
        ]

        movimiento, san = inferir_movimiento(
            self.board_logico, celdas_visitadas, energias
        )

        casillas = " - ".join(self.ultimas_casillas_visitadas)
        if movimiento is not None:
            self.ultimo_mensaje = f"Movimiento detectado: {san} ({movimiento.uci()})"
            print(
                f"♟️ {self.ultimo_mensaje} | celdas={casillas} | "
                f"energía warp={self.ultima_energia_warp:.3f}"
            )
        else:
            self.ultimo_mensaje = (
                f"Cambio detectado en {casillas}, sin jugada legal inferible"
            )
            print(
                f"⚠️ {self.ultimo_mensaje} | "
                f"energía warp={self.ultima_energia_warp:.3f}"
            )

        return movimiento, san

    def procesar_roi(self, roi_bgr: np.ndarray) -> Tuple[Optional[chess.Move], Optional[str]]:
        """Procesa un warp y devuelve una jugada solo al cerrar un ciclo válido."""
        warp_gray = self._normalizar_warp_gris(roi_bgr)
        rois, medias_actuales = self._extraer_celdas_warp_gris(warp_gray)
        if warp_gray is None or rois is None or medias_actuales is None:
            return None, None

        if not self.inicializado:
            self._inicializar(warp_gray)
            return None, None

        habia_celdas_no_limpias = not bool(np.all(self.estados_celdas == CELDA_LIMPIA))

        for indice in range(NUM_CELDAS_TABLERO):
            estado_actual = int(self.estados_celdas[indice])
            media_actual = float(medias_actuales[indice])
            media_anterior = float(self.medias_frame_anterior[indice])

            diferencia_media = (0.0
                if np.isnan(media_anterior)
                else abs(media_actual - media_anterior))
            self.ultima_diferencia_media_por_celda[indice] = diferencia_media

            if estado_actual == CELDA_LIMPIA:
                if (not np.isnan(media_anterior) and diferencia_media > self.umbral_diferencia_medias):
                    self.estados_celdas[indice] = CELDA_ALERTA
                    self.frames_pausa_por_celda[indice] = 0
                    self.muestras_medias_por_celda[indice].clear()
                    self.ultima_std_por_celda[indice] = 0.0
                    self.ciclo_tuvo_alerta = True
            else:
                self.ciclo_tuvo_alerta = True
                if estado_actual == CELDA_INTERRUMPIDA:
                    self.ciclo_tuvo_interrumpida = True

                if self.frames_pausa_por_celda[indice] < self.frames_pausa:
                    self.frames_pausa_por_celda[indice] += 1
                else:
                    muestras = self.muestras_medias_por_celda[indice]
                    muestras.append(media_actual)

                    if len(muestras) >= self.cantidad_muestras_std:
                        std_medias = float(np.std(np.asarray(muestras, dtype=np.float32)))
                        self.ultima_std_por_celda[indice] = std_medias

                        if std_medias > self.umbral_std:
                            self.estados_celdas[indice] = CELDA_INTERRUMPIDA
                            self.ciclo_tuvo_interrumpida = True
                        else:
                            self.estados_celdas[indice] = CELDA_LIMPIA

                        self.frames_pausa_por_celda[indice] = 0
                        muestras.clear()

            # Igual que en el notebook: siempre pasa a ser la media del frame
            # anterior para el próximo paso, independientemente del estado.
            self.medias_frame_anterior[indice] = media_actual

        todas_limpias = bool(np.all(self.estados_celdas == CELDA_LIMPIA))
        resultado: Tuple[Optional[chess.Move], Optional[str]] = (None, None)

        if todas_limpias:
            if habia_celdas_no_limpias or self.ciclo_tuvo_alerta:
                if self.ciclo_tuvo_interrumpida:
                    resultado = self._analizar_movimiento_entre_warps(warp_gray)
                else:
                    # El ciclo nunca alcanzó Interrumpida: falsa alarma.
                    self.ultima_energia_por_celda.fill(0.0)
                    self.ultima_energia_warp = 0.0
                    self.ultimas_celdas_visitadas.clear()
                    self.ultimas_casillas_visitadas.clear()
                    self.ultimo_mensaje = "Falsa alarma: el ciclo solo tuvo Alertas"

                self.ciclo_tuvo_alerta = False
                self.ciclo_tuvo_interrumpida = False

            # La referencia solo se reemplaza con las 64 celdas limpias y
            # después de resolver el ciclo que acaba de finalizar.
            self.warp_referencia_limpio = warp_gray.copy()

        return resultado

    def obtener_texto_progreso_celda(self, indice: int) -> str:
        estado = int(self.estados_celdas[indice])
        if estado == CELDA_LIMPIA:
            return ""

        espera_actual = int(self.frames_pausa_por_celda[indice])
        if espera_actual < self.frames_pausa:
            return f"P {espera_actual}/{self.frames_pausa}"

        tomadas = len(self.muestras_medias_por_celda[indice])
        return f"STD {tomadas}/{self.cantidad_muestras_std}"

    def generar_visualizacion(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Genera la ventana de estados equivalente a la del notebook."""
        warp_gray = self._normalizar_warp_gris(roi_bgr)
        if warp_gray is None:
            imagen = np.zeros((TAM_WARP + 120, TAM_WARP, 3), dtype=np.uint8)
            cv2.putText(imagen, "Detector inactivo", (280, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,)
            return imagen

        display = cv2.cvtColor(warp_gray, cv2.COLOR_GRAY2BGR)
        panel_h = 130
        salida = np.zeros((TAM_WARP + panel_h, TAM_WARP, 3), dtype=np.uint8)
        salida[:TAM_WARP] = display

        for fila in range(CELDAS_POR_LADO):
            for columna in range(CELDAS_POR_LADO):
                indice = fila * CELDAS_POR_LADO + columna
                estado = int(self.estados_celdas[indice])
                color = COLORES_ESTADO_CELDA[estado]
                x1 = columna * TAM_CELDA_WARP
                y1 = fila * TAM_CELDA_WARP
                x2 = x1 + TAM_CELDA_WARP
                y2 = y1 + TAM_CELDA_WARP

                cv2.rectangle(salida, (x1, y1), (x2 - 1, y2 - 1), color, 3)

                media = float(self.medias_frame_anterior[indice])
                texto_media = "Media: --" if np.isnan(media) else f"Media: {media:.1f}"
                cv2.rectangle(salida, (x1 + 3, y2 - 23), (x2 - 3, y2 - 3), (0, 0, 0), -1)
                cv2.putText(salida, texto_media, (x1 + 7, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1,)

                if estado != CELDA_LIMPIA:
                    cv2.rectangle(salida, (x1 + 3, y1 + 3), (x2 - 3, y1 + 43), (0, 0, 0), -1)
                    cv2.putText(salida, f"{indice_a_uci(indice)} {NOMBRES_ESTADO_CELDA[estado]}", (x1 + 7, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1,)
                    cv2.putText(salida, self.obtener_texto_progreso_celda(indice), (x1 + 7, y1 + 37), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,)

        referencia = ("ACTUALIZANDO" if np.all(self.estados_celdas == CELDA_LIMPIA) else "CONGELADA")
        casillas = " - ".join(self.ultimas_casillas_visitadas) or "(sin detección)"
        cv2.putText(salida, f"Referencia: {referencia} | Energia warp: {self.ultima_energia_warp:.3f}", (12, TAM_WARP + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,)
        cv2.putText(salida, f"Ultimas celdas: {casillas}", (12, TAM_WARP + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,)
        cv2.putText(salida, self.ultimo_mensaje[-105:] or "Esperando movimiento...", (12, TAM_WARP + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,)
        return salida


if __name__ == "__main__":
    # Ejecución independiente para probar únicamente parser + detector.
    # Mantiene las últimas esquinas durante oclusiones completas, igual que run.py.
    from parser_table import DetectorTablero

    max_frames_sin_cuadrilatero = 60
    cap = cv2.VideoCapture(1)
    parser = DetectorTablero(offset=80)
    detector_mov = DetectorMovimiento()
    prev_corners = None
    frames_sin_cuadrilatero = 0
    tracking_perdido = False

    cv2.namedWindow("Camara Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tablero Warpeado", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Estados de movimiento", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        parser.update_frame(frame)
        roi = None
        usando_fallback = False

        try:
            nuevas_corners = parser.detect_board_corners(prev_corners)
            if nuevas_corners is None:
                parser.reset()
                prev_corners = None
                frames_sin_cuadrilatero = 0
                if not tracking_perdido:
                    detector_mov.reiniciar_referencia_visual()
                tracking_perdido = True
            else:
                prev_corners = nuevas_corners
                roi = parser.get_board_roi()
                frames_sin_cuadrilatero = 0
                usando_fallback = bool(getattr(parser, "paciencia_oclusion", 0) > 0)
                tracking_perdido = False
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            if (prev_corners is not None and frames_sin_cuadrilatero < max_frames_sin_cuadrilatero):
                parser.esquinas = np.asarray(prev_corners, dtype=np.float32).copy()
                try:
                    roi = parser.get_board_roi()
                    frames_sin_cuadrilatero += 1
                    usando_fallback = True
                    tracking_perdido = False
                except (ValueError, cv2.error, np.linalg.LinAlgError):
                    roi = None

            if roi is None:
                parser.reset()
                prev_corners = None
                frames_sin_cuadrilatero = 0
                if not tracking_perdido:
                    detector_mov.reiniciar_referencia_visual()
                tracking_perdido = True

        if roi is not None:
            movimiento, san = detector_mov.procesar_roi(roi)
            if movimiento is not None:
                cv2.putText(roi, f"Jugada: {san}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3,)

            cv2.imshow("Tablero Warpeado", roi)
            cv2.imshow("Estados de movimiento", detector_mov.generar_visualizacion(roi))

        frame_mostrado = frame.copy()
        if usando_fallback:
            cv2.putText(frame_mostrado, "Oclusion: usando ultimas esquinas", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2,)

        cv2.imshow("Camara Original", frame_mostrado)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("r"), ord("R")):
            detector_mov.reiniciar_referencia_visual()

    cap.release()
    cv2.destroyAllWindows()
