import chess
import chess.engine
import threading
import os

_DIR_UI   = os.path.dirname(os.path.abspath(__file__))
_DIR_RAIZ = os.path.dirname(os.path.dirname(_DIR_UI))
_EXE_DEFAULT = os.path.join(_DIR_RAIZ, "models", "stockfish", "stockfish-windows-x86-64-avx2.exe")


class StockfishAdvisor:
    """Analiza el mejor movimiento para cada bando en un hilo separado.

    Después de cada jugada detectada, recibe una copia del chess.Board y
    lanza un análisis de Stockfish (con límite de tiempo corto) que corre
    sin bloquear el loop de captura.  El resultado queda disponible en
    self.sugerencias = {'blancas': str|None, 'negras': str|None}.
    """

    def __init__(self, path_exe=None, tiempo_ms=500):
        path_exe = path_exe or _EXE_DEFAULT
        self.tiempo_ms = tiempo_ms
        self.engine = chess.engine.SimpleEngine.popen_uci(path_exe)
        self.sugerencias = {'blancas': None, 'negras': None}
        self._lock    = threading.Lock()
        self._thread  = None

    # ── API pública ────────────────────────────────────────────────────────────

    def analizar_async(self, board: chess.Board):
        """Lanza el análisis en background.  Devuelve inmediatamente."""
        if self._thread and self._thread.is_alive():
            return  # análisis anterior todavía en curso
        self._thread = threading.Thread(target=self._worker, args=(board.copy(),), daemon=True)
        self._thread.start()

    def get_sugerencias(self):
        with self._lock:
            return dict(self.sugerencias)

    def cerrar(self):
        try:
            self.engine.quit()
        except Exception:
            pass

    # ── Hilo de trabajo ────────────────────────────────────────────────────────

    @staticmethod
    def _board_para_color(board: chess.Board, color: chess.Color) -> chess.Board:
        """Devuelve un chess.Board con el turno forzado a `color`.

        Si el turno ya coincide, devuelve una copia directa.  Si no, reconstruye
        el board desde un FEN modificado para que python-chess y Stockfish estén
        de acuerdo en quién mueve.  También limpia el en-passant cuando se
        invierte el turno (era válido solo para el bando original).
        """
        if board.turn == color:
            return board.copy()
        parts    = board.fen().split()
        parts[1] = 'w' if color == chess.WHITE else 'b'
        parts[3] = '-'   # en passant solo aplica al turno original
        return chess.Board(' '.join(parts))

    def _worker(self, board: chess.Board):
        sug = {}
        for color, key in [(chess.WHITE, 'blancas'), (chess.BLACK, 'negras')]:
            b = self._board_para_color(board, color)
            if not any(True for _ in b.legal_moves):
                sug[key] = None
                continue
            try:
                result = self.engine.analyse(
                    b, chess.engine.Limit(time=self.tiempo_ms / 1000)
                )
                pv   = result.get('pv', [])
                best = pv[0] if pv else None
                if best is None:
                    sug[key] = None
                    continue

                san   = b.san(best)
                score = result['score']

                if score.is_mate():
                    n        = score.white().mate() if color == chess.WHITE else score.black().mate()
                    sug[key] = f"{san}  M{abs(n)}"
                else:
                    cp       = score.white().score() if color == chess.WHITE else score.black().score()
                    sug[key] = f"{san}  {cp:+d}" if cp is not None else san

            except Exception:
                sug[key] = None

        with self._lock:
            self.sugerencias.update(sug)
