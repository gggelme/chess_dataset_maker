
# peon
class Pawn: 
    def __init__(self, color, row, col):
        self.color = color
        self.row = row
        self.col = col
        self.primer_mov = True
        self.valor = 1 if color == 'blanco' else -1 
        
    def obtener_movimientos_validos(self, tablero):
        movimientos = []
        direccion = 1 if self.color == 'blanco' else -1
        
        # omito que se vaya del tablero
        # avanzar un paso 
        if tablero[self.row + direccion][self.col] == 0: # 0 = celda vacía
            movimientos.append((self.row + direccion, self.col))
            
            # avanzar dos pasos si es su primer movimiento
            if self.primer_mov and tablero[self.row + (2 * direccion)][self.col] == 0:
                movimientos.append((self.row + (2 * direccion), self.col))
        
        # comer en diagonal

        # primero tenemos que ver como representamos el tablero
        # blancas positivos y negras negativas ?
        # 1 peon, 2 caballo, 3 alfil... ??

        return movimientos

    def mover(self, nueva_row, nueva_col):
        self.row = nueva_row
        self.col = nueva_col
        self.primer_mov = False

    def __repr__(self):
        return f"Peon({self.color}) en [{self.row},{self.col}]"
    


# ???


# alfil
class Bishop:
    def __init__(self, color, row, col):
        self.color = color
        self.row = row
        self.col = col
        self.valor = 3 if color == 'blanco' else -3 

    def obtener_movimientos_validos(self, tablero):
        movimientos = []
        direcciones = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for d_row, d_col in direcciones:
            r = self.row + d_row
            c = self.col + d_col
            
            # mientras no nos salgamos del tablero
            while 0 <= r <= 7 and 0 <= c <= 7:
                contenido_celda = tablero[r][c]
                
                if contenido_celda == 0:
                    movimientos.append((r, c))
                else:
                    if contenido_celda.color != self.color:
                        movimientos.append((r, c))
                    
                    break 
                    
                r += d_row
                c += d_col

        return movimientos

    def mover(self, nueva_row, nueva_col):
        self.row = nueva_row
        self.col = nueva_col

    def __repr__(self):
        return f"Alfil({self.color}) en [{self.row},{self.col}]"