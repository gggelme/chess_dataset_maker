import json
import os
import chess

class GameLogger:
    """Persiste la partida en archivos JSON separados por bando y en un historial completo."""

    def __init__(self, dir_log: str):
        self.dir_log = dir_log
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
        self.historial_NAL = []

        for color in (chess.WHITE, chess.BLACK):
            self._guardar(color)

    def registrar(self, color: chess.Color, san: str, uci: str, recomendado: 'str | None'):
        """Agrega un movimiento a los JSON correspondientes."""
        self._contadores[color] += 1
        entrada = {
            "numero":      self._contadores[color],
            "jugada":      san,
            "uci":         uci,
            "recomendado": recomendado,
        }
        self._data[color]["movimientos"].append(entrada)
        self._guardar(color)

        # Conserva el formato "e2-e4" transformando el string UCI ("e2e4")
        if len(uci) >= 4:
            self.historial_NAL.append(f"{uci[:2]}-{uci[2:4]}")

        print(f"  [log] {'blancas' if color == chess.WHITE else 'negras'} "
              f"#{self._contadores[color]}: {san} (rec: {recomendado})")

    def _guardar(self, color: chess.Color):
        with open(self._paths[color], 'w', encoding='utf-8') as f:
            json.dump(self._data[color], f, ensure_ascii=False, indent=2)

    def guardar_historial_completo(self, nombre_archivo="partida.json"):
        """Guarda el historial completo acumulado (reemplaza a board.py)."""
        ruta_completa = os.path.join(self.dir_log, nombre_archivo)
        with open(ruta_completa, "w", encoding="utf-8") as f:
            json.dump({"movimientos": self.historial_NAL}, f, indent=4)
        print(f"Historial completo guardado en: {ruta_completa}")