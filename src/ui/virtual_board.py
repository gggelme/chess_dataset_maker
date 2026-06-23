import os
import pygame
import numpy as np

# mapea una matriz de 8x8 en un tablero de ajedrez digital
# python src/ui/virtual_board.py
class LiveBoard:
    def __init__(self, tamano_celda=80):
        self.tamano_celda = tamano_celda
        self.ancho_tablero = tamano_celda * 8
        self.corriendo = True
        
        # Configuración de rutas
        self.dir_actual = os.path.dirname(os.path.abspath(__file__))
        self.dir_raiz = os.path.dirname(os.path.dirname(self.dir_actual))
        self.carpeta_assets = os.path.join(self.dir_raiz, "assets")
        self.carpeta_piezas = os.path.join(self.carpeta_assets, "pieces")
        
        self.mapeo_piezas = {
            1: 'wP.png',  2: 'wN.png',  3: 'wB.png',  4: 'wR.png',  5: 'wK.png',  6: 'wQ.png',
           -1: 'bP.png', -2: 'bN.png', -3: 'bB.png', -4: 'bR.png', -5: 'bK.png', -6: 'bQ.png',
        }
        
        # Inicialización de Pygame
        pygame.init()
        self.pantalla = pygame.display.set_mode((self.ancho_tablero, self.ancho_tablero))
        pygame.display.set_caption("Tablero Digital en Tiempo Real")
        self.assets = self._cargar_assets()

    def _cargar_assets(self):
        """Método privado para cargar las imágenes una sola vez al instanciar la clase."""
        assets = {}
        # piezas
        for valor, archivo in self.mapeo_piezas.items():
            ruta = os.path.join(self.carpeta_piezas, archivo)
            img = pygame.image.load(ruta).convert_alpha()
            assets[valor] = pygame.transform.smoothscale(img, (self.tamano_celda, self.tamano_celda))
            
        # fondo completo
        ruta_fondo = os.path.join(self.carpeta_piezas, 'brown.png')
        fondo = pygame.image.load(ruta_fondo).convert()
        assets['fondo'] = pygame.transform.smoothscale(fondo, (self.ancho_tablero, self.ancho_tablero))

        return assets

    def actualizar(self, matriz):
        """
        Recibe la matriz actualizada, procesa la ventana y la dibuja.
        Retorna True si la ventana sigue abierta, False si se cerró.
        """
        if not isinstance(matriz, np.ndarray):
            raise TypeError(f"Se esperaba un numpy.ndarray, pero se recibió {type(matriz).__name__}. "
                            "Asegurate de convertir tu lista usando np.array(tu_lista).")
        
        # -------------------------
        
        if not self.corriendo:
            return False

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                self.corriendo = False
                pygame.quit()
                return False

        # 1. Dibujar TODO el tablero de fondo de una sola vez
        self.pantalla.blit(self.assets['fondo'], (0, 0))

        # 2. Dibujar únicamente las piezas donde corresponda
        for fila in range(8):
            for col in range(8):
                pieza = matriz[fila, col]
                if pieza != 0:
                    x = col * self.tamano_celda
                    y = fila * self.tamano_celda
                    self.pantalla.blit(self.assets[pieza], (x, y))

        pygame.display.flip()
        return True

# prueba para ver si funciona, las matrices son de prueba
if __name__ == "__main__":
    # negras (negativo) arriba — blancas (positivo) abajo
    m1 = [
    [-4, -2, -3, -6, -5, -3, -2, -4],  # fila 0 – back rank negras
    [-1, -1, -1, -1, -1, -1, -1, -1],  # fila 1 – peones negros
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 1,  1,  1,  1,  1,  1,  1,  1],  # fila 6 – peones blancos
    [ 4,  2,  3,  6,  5,  3,  2,  4],] # fila 7 – back rank blancas

    m2 = [
    [-4, -2, -3, -6, -5, -3, -2, -4],
    [-1, -1, -1,  0, -1, -1, -1, -1],
    [ 0,  0,  0, -1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 1,  1,  1,  1,  1,  1,  1,  1],
    [ 4,  2,  3,  6,  5,  3,  2,  4],]

    m3 = [
    [-4, -2, -3, -6, -5, -3, -2, -4],
    [-1, -1, -1,  0, -1, -1, -1, -1],
    [ 0,  0,  0, -1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  1,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 1,  0,  1,  1,  1,  1,  1,  1],
    [ 4,  2,  3,  6,  5,  3,  2,  4],]

    lista_matrices = [np.array(m1), np.array(m2), np.array(m3)]
    
    tablero = LiveBoard()
    
    indice_matriz = 0
    ultimo_cambio = pygame.time.get_ticks() # Registra el tiempo inicial en milisegundos

    while True:
        tiempo_actual = pygame.time.get_ticks()
        
        # Si pasaron 3000 milisegundos (3 segundos), avanzamos a la siguiente matriz
        if tiempo_actual - ultimo_cambio >= 3000:
            indice_matriz = (indice_matriz + 1) % len(lista_matrices)
            ultimo_cambio = tiempo_actual

        # Pasamos la matriz actual al tablero
        if not tablero.actualizar(lista_matrices[indice_matriz]):
            break
            
        # Pequeña pausa
        pygame.time.wait(30)