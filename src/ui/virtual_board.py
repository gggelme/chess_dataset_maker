import os
import pygame
import numpy as np

# python src/ui/virtual_board.py

class LiveBoard:
    def __init__(self, tamano_celda=80):
        self.tamano_celda = tamano_celda
        self.ancho_tablero = tamano_celda * 8
        self.ancho_panel = 250  # Espacio para el reloj y turnos
        self.corriendo = True
        
        # Variables de estado del juego
        self.turno = 1  # 1 (Blancas) o -1 (Negras)
        self.tiempos = {1: 0.0, -1: 0.0}
        self.matriz_anterior = None
        
        # Configuración de rutas
        self.dir_actual = os.path.dirname(os.path.abspath(__file__))
        self.dir_raiz = os.path.dirname(os.path.dirname(self.dir_actual))
        self.carpeta_assets = os.path.join(self.dir_raiz, "assets")
        self.carpeta_piezas = os.path.join(self.carpeta_assets, "pieces")
        
        self.mapeo_piezas = {
            1: 'wP.png',  2: 'wN.png',  3: 'wB.png',  4: 'wR.png',  5: 'wK.png',  6: 'wQ.png',
           -1: 'bP.png', -2: 'bN.png', -3: 'bB.png', -4: 'bR.png', -5: 'bK.png', -6: 'bQ.png',
        }
        
        # Sugerencias de Stockfish (actualizar desde fuera)
        self.sugerencias = {'blancas': None, 'negras': None}

        # Inicialización
        pygame.init()
        pygame.font.init()
        self.fuente        = pygame.font.SysFont("Verdana", 22, bold=True)
        self.fuente_chica  = pygame.font.SysFont("Verdana", 18)
        self.fuente_mini   = pygame.font.SysFont("Verdana", 14)

        # Pantalla más ancha para incluir el panel
        self.pantalla = pygame.display.set_mode((self.ancho_tablero + self.ancho_panel, self.ancho_tablero))
        pygame.display.set_caption("Tablero en Tiempo Real")
        self.assets = self._cargar_assets()

        self.ultimo_tick = pygame.time.get_ticks()


    def _inferir_turno(self, matriz_nueva):
        """Infiere el turno viendo qué pieza desapareció de su origen."""
        if self.matriz_anterior is not None:
            # Buscamos la coordenada donde antes había una pieza y ahora hay un 0
            cambios = np.where((self.matriz_anterior != 0) & (matriz_nueva == 0))
            if len(cambios[0]) > 0:
                pieza_movida = self.matriz_anterior[cambios[0][0], cambios[1][0]]
                # Si movió una blanca (>0), pasa el turno a negras (-1) y viceversa
                self.turno = -1 if pieza_movida > 0 else 1
                
        self.matriz_anterior = matriz_nueva.copy()


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


    def _dibujar_panel(self):
        COLOR_FONDO   = (30, 33, 40) # Un tono más oscuro y elegante
        COLOR_SUBTIT  = (140, 145, 155)
        COLOR_BLANCAS = (255, 210, 80)
        COLOR_NEGRAS  = (90, 170, 255)
        
        x0 = self.ancho_tablero
        rect_panel = pygame.Rect(x0, 0, self.ancho_panel, self.ancho_tablero)
        pygame.draw.rect(self.pantalla, COLOR_FONDO, rect_panel)

        # ── Control de Tiempo ──────────────────────────────────────────────
        self.pantalla.blit(self.fuente_chica.render("CONTROL DE TIEMPO", True, COLOR_SUBTIT), (x0 + 20, 20))

        def dibujar_reloj(y, titulo, tiempo, activo):
            color_caja  = (46, 160, 67) if activo else (45, 48, 56)
            color_texto = (255, 255, 255) if activo else (180, 185, 195)
            
            caja = pygame.Rect(x0 + 20, y, 210, 65)
            pygame.draw.rect(self.pantalla, color_caja, caja, border_radius=8)
            
            m, s = divmod(int(tiempo), 60)
            self.pantalla.blit(self.fuente_chica.render(titulo, True, color_texto), (caja.x + 15, caja.y + 10))
            self.pantalla.blit(self.fuente.render(f"{m:02d}:{s:02d}", True, color_texto), (caja.x + 15, caja.y + 32))

        dibujar_reloj(55, "Blancas", self.tiempos[1],  self.turno == 1)
        dibujar_reloj(135, "Negras", self.tiempos[-1], self.turno == -1)

        # ── Separador y Mejor Movimiento ───────────────────────────────────
        sep_y = 230
        pygame.draw.line(self.pantalla, (55, 60, 70), (x0 + 20, sep_y), (x0 + 230, sep_y), 2)
        
        self.pantalla.blit(self.fuente_chica.render("MEJOR MOVIMIENTO", True, COLOR_SUBTIT), (x0 + 20, sep_y + 15))

        def dibujar_sugerencia(y, titulo, texto, color_acento):
            caja = pygame.Rect(x0 + 20, y, 210, 60)
            pygame.draw.rect(self.pantalla, (35, 38, 45), caja, border_radius=8)
            pygame.draw.rect(self.pantalla, color_acento, pygame.Rect(caja.x, caja.y, 4, caja.height), border_radius=4)

            self.pantalla.blit(self.fuente_mini.render(titulo, True, color_acento), (caja.x + 15, caja.y + 10))

            if texto:
                txt_render = self.fuente.render(texto, True, (220, 220, 220))
            else:
                txt_render = self.fuente_chica.render("analizando...", True, (100, 105, 115))

            self.pantalla.blit(txt_render, (caja.x + 15, caja.y + 28))

        sug = self.sugerencias
        dibujar_sugerencia(sep_y + 45, "BLANCAS", sug.get('blancas'), COLOR_BLANCAS)
        dibujar_sugerencia(sep_y + 115, "NEGRAS",  sug.get('negras'),  COLOR_NEGRAS)


    def actualizar(self, matriz):
        if not isinstance(matriz, np.ndarray):
            raise TypeError("Se esperaba un numpy.ndarray.")
        
        if not self.corriendo: return False

        # Actualizar reloj
        ahora = pygame.time.get_ticks()
        dt = (ahora - self.ultimo_tick) / 1000.0  
        self.ultimo_tick = ahora
        
        # Sumamos el tiempo al jugador que tiene el turno
        self.tiempos[self.turno] += dt

        # Inferir si hubo cambio de turno
        self._inferir_turno(matriz)

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                self.corriendo = False
                pygame.quit()
                return False

        # Dibujado general
        self.pantalla.blit(pygame.transform.flip(self.assets['fondo'], True, False), (0, 0))
        
        # Dibujar piezas (usando np.nonzero para mayor eficiencia)
        filas, cols = np.nonzero(matriz)
        for f, c in zip(filas, cols):
            pieza = matriz[f, c]
            self.pantalla.blit(self.assets[pieza], (c * self.tamano_celda, f * self.tamano_celda))

        # Dibujar el panel UI
        self._dibujar_panel()

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