import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pieces import Pawn, Knight, Bishop, Rook, King, Queen


class Cell:

    def __init__(self, x_left, x_right, y_up, y_down, row=None, col=None):
        self.x_left = x_left
        self.x_right = x_right
        self.y_up = y_up
        self.y_down = y_down
        self.row = row
        self.col = col
        self.value = 0

    def __repr__(self):
        return (f"Cell[{self.row},{self.col}] "
                f"x=({self.x_left},{self.x_right}) y=({self.y_up},{self.y_down}) "
                f"value={self.value}")


class Board:

    # peon=1  caballo=2  alfil=3  torre=4  rey=5  reina=6
    _CLASE_POR_VALOR = {
        1: Pawn, 2: Knight, 3: Bishop,
        4: Rook, 5: King,  6: Queen,
    }

    def __init__(self, nueva_partida: bool, imagen_rectificada=None,
                 y_pos=None, x_pos=None):
        self.imagen = imagen_rectificada  # imagen HSV (puede ser None)
        self.y_pos  = y_pos               # 9 límites de filas   (Hough)
        self.x_pos  = x_pos               # 9 límites de columnas (Hough)

        if imagen_rectificada is not None:
            lado = imagen_rectificada.shape[0]
            self.tam_celda = lado / 8
            self.celdas = self._construir_celdas()
        else:
            self.tam_celda = None
            self.celdas = None

        self.matriz = np.zeros((8, 8), dtype=int)
        self.piezas = [[None] * 8 for _ in range(8)]

        if nueva_partida:
            self.posicion_inicial = [
                [-4, -2, -3, -6, -5, -3, -2, -4],  # fila 0 – back rank negras (arriba)
                [-1, -1, -1, -1, -1, -1, -1, -1],  # fila 1 – peones negros
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0,  0],
                [ 1,  1,  1,  1,  1,  1,  1,  1],  # fila 6 – peones blancos
                [ 4,  2,  3,  6,  5,  3,  2,  4],  # fila 7 – back rank blancas (abajo)
            ]
            self._inicializar_piezas()
        # else: partida en curso → lógica de detección por imagen (a implementar)

    def _inicializar_piezas(self):
        for fila in range(8):
            for col in range(8):
                valor = self.posicion_inicial[fila][col]
                self.matriz[fila][col] = valor
                if valor != 0:
                    color = 'blanco' if valor > 0 else 'negro'
                    clase = self._CLASE_POR_VALOR[abs(valor)]
                    self.piezas[fila][col] = clase(color, fila, col)

    def _construir_celdas(self):
        celdas = []
        usar_hough = self.y_pos is not None and self.x_pos is not None
        for fila in range(8):
            fila_celdas = []
            for col in range(8):
                if usar_hough:
                    y1, y2 = self.y_pos[fila], self.y_pos[fila + 1]
                    x1, x2 = self.x_pos[col],  self.x_pos[col  + 1]
                else:
                    x1 = int(col  * self.tam_celda)
                    y1 = int(fila * self.tam_celda)
                    x2 = int((col  + 1) * self.tam_celda)
                    y2 = int((fila + 1) * self.tam_celda)
                celda = Cell(x1, x2, y1, y2, row=fila, col=col)
                fila_celdas.append(celda)
            celdas.append(fila_celdas)
        return celdas

    def get_imagen_celda(self, fila, col):
        """Sub-imagen HSV de la celda [fila][col]."""
        c = self.celdas[fila][col]
        return self.imagen[c.y_up:c.y_down, c.x_left:c.x_right]

    def dibujar(self, img=None):
        import cv2
        if img is None:
            img = self.imagen.copy()
        for fila in self.celdas:
            for celda in fila:
                cv2.rectangle(
                    img,
                    (celda.x_left, celda.y_up),
                    (celda.x_right, celda.y_down),
                    (255, 0, 0),
                    2
                )
        return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    def _repr_(self):
        return self.matriz