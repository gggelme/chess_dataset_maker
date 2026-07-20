import json
import os
import chess


class GameLogger:
    """Persiste la partida en dos archivos JSON separados por bando.

    data/log/blancas.json  y  data/log/negras.json

    Cada movimiento queda registrado con:
      numero      : ordinal del turno de ese bando (1, 2, 3, ...)
      jugada      : notación SAN  (e4, Nf3, O-O, ...)
      uci         : notación UCI  (e2e4, g1f3, e1g1, ...)
      recomendado : sugerencia de Stockfish vigente ANTES de la jugada  (o null)
    """

    def __init__(self, dir_log: str):
        os.makedirs(dir_log, exist_ok=True)
        self._paths = {
            chess.WHITE: os.path.join(dir_log, 'blancas.json'),
            chess.BLACK: os.path.join(dir_log, 'negras.json'),
        }
        self._data = {
            chess.WHITE: {"jugador": "blancas", "movimientos": []},
            chess.BLACK: {"jugador": "negras",  "movimientos": []},
        }
        self._contadores = {chess.WHITE: 0, chess.BLACK: 0}

        # Persistir archivos vacíos al arrancar para que existan desde el inicio
        for color in (chess.WHITE, chess.BLACK):
            self._guardar(color)

    def registrar(self, color: chess.Color, san: str, uci: str,
                  recomendado: 'str | None'):
        """Agrega un movimiento al JSON del bando y lo persiste inmediatamente."""
        self._contadores[color] += 1
        entrada = {
            "numero":      self._contadores[color],
            "jugada":      san,
            "uci":         uci,
            "recomendado": recomendado,
        }
        self._data[color]["movimientos"].append(entrada)
        self._guardar(color)
        print(f"  [log] {'blancas' if color == chess.WHITE else 'negras'} "
              f"#{self._contadores[color]}: {san} (rec: {recomendado})")

    def _guardar(self, color: chess.Color):
        with open(self._paths[color], 'w', encoding='utf-8') as f:
            json.dump(self._data[color], f, ensure_ascii=False, indent=2)
