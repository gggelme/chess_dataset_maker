
# Convención:
#   blanco → valor positivo, avanza hacia filas crecientes (+1)
#   negro  → valor negativo, avanza hacia filas decrecientes (-1)
#
# peon=1  caballo=2  alfil=3  torre=4  rey=5  reina=6


def _mover(pieza, nueva_row, nueva_col, piezas):
    """Lógica común de validación, captura y actualización de grilla."""
    movimientos = pieza.obtener_movimientos_validos(piezas)
    if (nueva_row, nueva_col) not in movimientos:
        raise ValueError(
            f"{pieza} no puede moverse a [{nueva_row},{nueva_col}]. "
            f"Movimientos válidos: {movimientos}"
        )
    pieza_comida = piezas[nueva_row][nueva_col]
    piezas[pieza.row][pieza.col] = None
    piezas[nueva_row][nueva_col] = pieza
    pieza.row = nueva_row
    pieza.col = nueva_col
    return pieza_comida  # None si la celda estaba vacía


# ── Peón ─────────────────────────────────────────────────────────────────────

class Pawn:
    def __init__(self, color, row, col):
        self.color = color
        self.row = row
        self.col = col
        self.primer_mov = True
        self.valor = 1 if color == 'blanco' else -1

    def obtener_movimientos_validos(self, piezas, en_passant_col=None):
        movimientos = []
        direccion = 1 if self.color == 'blanco' else -1
        nueva_fila = self.row + direccion

        # avance recto (solo a casilla vacía)
        if 0 <= nueva_fila <= 7 and piezas[nueva_fila][self.col] is None:
            movimientos.append((nueva_fila, self.col))
            dos_filas = self.row + 2 * direccion
            if self.primer_mov and 0 <= dos_filas <= 7 and piezas[dos_filas][self.col] is None:
                movimientos.append((dos_filas, self.col))

        # capturas diagonales normales + en passant
        for d_col in (-1, 1):
            c = self.col + d_col
            if 0 <= nueva_fila <= 7 and 0 <= c <= 7:
                objetivo = piezas[nueva_fila][c]
                if objetivo is not None and objetivo.color != self.color:
                    movimientos.append((nueva_fila, c))
                elif en_passant_col == c:
                    # el peón enemigo está en (self.row, c) pero capturamos yendo a (nueva_fila, c)
                    movimientos.append((nueva_fila, c))

        return movimientos

    def mover(self, nueva_row, nueva_col, piezas, en_passant_col=None, promocion=None):
        movimientos = self.obtener_movimientos_validos(piezas, en_passant_col)
        if (nueva_row, nueva_col) not in movimientos:
            raise ValueError(
                f"{self} no puede moverse a [{nueva_row},{nueva_col}]. "
                f"Movimientos válidos: {movimientos}"
            )

        # en passant: movimiento diagonal hacia casilla vacía
        es_en_passant = (nueva_col != self.col) and (piezas[nueva_row][nueva_col] is None)

        if es_en_passant:
            pieza_comida = piezas[self.row][nueva_col]
            piezas[self.row][nueva_col] = None
        else:
            pieza_comida = piezas[nueva_row][nueva_col]

        piezas[self.row][self.col] = None
        piezas[nueva_row][nueva_col] = self
        self.row = nueva_row
        self.col = nueva_col
        self.primer_mov = False

        # Promoción: peón llega a la última fila del bando rival
        fila_final = 7 if self.color == 'blanco' else 0
        if self.row == fila_final:
            clase = promocion if promocion is not None else Queen
            nueva_pieza = clase(self.color, self.row, self.col)
            piezas[self.row][self.col] = nueva_pieza
            self._promovido = True
            print(f"{self} fue promovido a {nueva_pieza}")

        return pieza_comida

    def __del__(self):
        if not getattr(self, '_promovido', False):
            print(f"{self} fue eliminado del tablero")

    def __repr__(self):
        return f"Peon({self.color}) en [{self.row},{self.col}]"


# ── Caballo ───────────────────────────────────────────────────────────────────

class Knight:
    def __init__(self, color, row, col):
        self.color = color
        self.row = row
        self.col = col
        self.valor = 2 if color == 'blanco' else -2

    def obtener_movimientos_validos(self, piezas):
        movimientos = []
        saltos = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  ( 1, -2), ( 1, 2), ( 2, -1), ( 2,  1)]
        for d_row, d_col in saltos:
            r, c = self.row + d_row, self.col + d_col
            if 0 <= r <= 7 and 0 <= c <= 7:
                objetivo = piezas[r][c]
                if objetivo is None or objetivo.color != self.color:
                    movimientos.append((r, c))
        return movimientos

    def mover(self, nueva_row, nueva_col, piezas):
        return _mover(self, nueva_row, nueva_col, piezas)

    def __del__(self):
        print(f"{self} fue eliminado del tablero")

    def __repr__(self):
        return f"Caballo({self.color}) en [{self.row},{self.col}]"


# ── Alfil ─────────────────────────────────────────────────────────────────────

class Bishop:
    def __init__(self, color, row, col):
        self.color = color
        self.row = row
        self.col = col
        self.valor = 3 if color == 'blanco' else -3

    def obtener_movimientos_validos(self, piezas):
        movimientos = []
        for d_row, d_col in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            r, c = self.row + d_row, self.col + d_col
            while 0 <= r <= 7 and 0 <= c <= 7:
                objetivo = piezas[r][c]
                if objetivo is None:
                    movimientos.append((r, c))
                else:
                    if objetivo.color != self.color:
                        movimientos.append((r, c))
                    break
                r += d_row
                c += d_col
        return movimientos

    def mover(self, nueva_row, nueva_col, piezas):
        return _mover(self, nueva_row, nueva_col, piezas)

    def __del__(self):
        print(f"{self} fue eliminado del tablero")

    def __repr__(self):
        return f"Alfil({self.color}) en [{self.row},{self.col}]"


# ── Torre ─────────────────────────────────────────────────────────────────────

class Rook:
    def __init__(self, color, row, col):
        self.color = color
        self.row = row
        self.col = col
        self.primer_mov = True
        self.valor = 4 if color == 'blanco' else -4

    def obtener_movimientos_validos(self, piezas):
        movimientos = []
        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = self.row + d_row, self.col + d_col
            while 0 <= r <= 7 and 0 <= c <= 7:
                objetivo = piezas[r][c]
                if objetivo is None:
                    movimientos.append((r, c))
                else:
                    if objetivo.color != self.color:
                        movimientos.append((r, c))
                    break
                r += d_row
                c += d_col
        return movimientos

    def mover(self, nueva_row, nueva_col, piezas):
        comida = _mover(self, nueva_row, nueva_col, piezas)
        self.primer_mov = False
        return comida

    def __del__(self):
        print(f"{self} fue eliminado del tablero")

    def __repr__(self):
        return f"Torre({self.color}) en [{self.row},{self.col}]"


# ── Rey ───────────────────────────────────────────────────────────────────────

class King:
    def __init__(self, color, row, col):
        self.color = color
        self.row = row
        self.col = col
        self.primer_mov = True
        self.valor = 5 if color == 'blanco' else -5

    def obtener_movimientos_validos(self, piezas):
        movimientos = []
        for d_row, d_col in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            r, c = self.row + d_row, self.col + d_col
            if 0 <= r <= 7 and 0 <= c <= 7:
                objetivo = piezas[r][c]
                if objetivo is None or objetivo.color != self.color:
                    movimientos.append((r, c))

        # Enroque (solo si el rey no se movió)
        if self.primer_mov:
            fila = self.row

            # Enroque corto: torre en col 7, cols 5 y 6 vacías
            torre_k = piezas[fila][7]
            if (isinstance(torre_k, Rook) and torre_k.color == self.color
                    and torre_k.primer_mov
                    and piezas[fila][5] is None and piezas[fila][6] is None):
                movimientos.append((fila, 6))

            # Enroque largo: torre en col 0, cols 1, 2 y 3 vacías
            torre_q = piezas[fila][0]
            if (isinstance(torre_q, Rook) and torre_q.color == self.color
                    and torre_q.primer_mov
                    and piezas[fila][1] is None
                    and piezas[fila][2] is None
                    and piezas[fila][3] is None):
                movimientos.append((fila, 2))

        return movimientos

    def mover(self, nueva_row, nueva_col, piezas):
        movimientos = self.obtener_movimientos_validos(piezas)
        if (nueva_row, nueva_col) not in movimientos:
            raise ValueError(
                f"{self} no puede moverse a [{nueva_row},{nueva_col}]. "
                f"Movimientos válidos: {movimientos}"
            )

        # Detectar enroque: el rey se desplaza 2 columnas
        if abs(nueva_col - self.col) == 2:
            fila = self.row
            if nueva_col == 6:          # enroque corto
                torre = piezas[fila][7]
                piezas[fila][7] = None
                piezas[fila][5] = torre
                torre.col = 5
            else:                       # enroque largo (nueva_col == 2)
                torre = piezas[fila][0]
                piezas[fila][0] = None
                piezas[fila][3] = torre
                torre.col = 3
            torre.primer_mov = False

        piezas[self.row][self.col] = None
        piezas[nueva_row][nueva_col] = self
        self.row = nueva_row
        self.col = nueva_col
        self.primer_mov = False
        return None  # el enroque no captura ninguna pieza

    def __del__(self):
        print(f"{self} fue eliminado del tablero")

    def __repr__(self):
        return f"Rey({self.color}) en [{self.row},{self.col}]"


# ── Reina ─────────────────────────────────────────────────────────────────────

class Queen:
    def __init__(self, color, row, col):
        self.color = color
        self.row = row
        self.col = col
        self.valor = 6 if color == 'blanco' else -6

    def obtener_movimientos_validos(self, piezas):
        movimientos = []
        for d_row, d_col in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            r, c = self.row + d_row, self.col + d_col
            while 0 <= r <= 7 and 0 <= c <= 7:
                objetivo = piezas[r][c]
                if objetivo is None:
                    movimientos.append((r, c))
                else:
                    if objetivo.color != self.color:
                        movimientos.append((r, c))
                    break
                r += d_row
                c += d_col
        return movimientos

    def mover(self, nueva_row, nueva_col, piezas):
        return _mover(self, nueva_row, nueva_col, piezas)

    def __del__(self):
        print(f"{self} fue eliminado del tablero")

    def __repr__(self):
        return f"Reina({self.color}) en [{self.row},{self.col}]"
