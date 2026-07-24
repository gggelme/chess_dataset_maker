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

        ruta_font_med = os.path.join(self.carpeta_assets, "menu", "Montserrat-Medium.ttf")
        ruta_font_bold = os.path.join(self.carpeta_assets, "menu", "Montserrat-Bold.ttf")

        self.fuente        = pygame.font.Font(ruta_font_bold, 22)
        self.fuente_chica  = pygame.font.Font(ruta_font_med, 18)
        self.fuente_chica_bold = pygame.font.Font(ruta_font_bold, 18)
        self.fuente_mini   = pygame.font.Font(ruta_font_med, 14)

        self.fuente_grande = pygame.font.Font(ruta_font_bold, 36)

        self.resultado_final = None # Guardará '1-0', '0-1' o '1/2-1/2'

        # Pantalla más ancha para incluir el panel
        self.pantalla = pygame.display.set_mode((self.ancho_tablero + self.ancho_panel, self.ancho_tablero))
        pygame.display.set_caption("Tablero en Tiempo Real")
        self.assets = self._cargar_assets()

        self.ultimo_tick = pygame.time.get_ticks()
        self.modo_visor = False


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
        COLOR_FONDO   = (17, 31, 57)
        COLOR_SUBTIT  = (140, 145, 155)
        COLOR_BLANCAS = (255, 210, 80)
        COLOR_NEGRAS  = (90, 170, 255)
        
        x0 = self.ancho_tablero
        rect_panel = pygame.Rect(x0, 0, self.ancho_panel, self.ancho_tablero)
        pygame.draw.rect(self.pantalla, COLOR_FONDO, rect_panel)

        # ── Control de Tiempo ──────────────────────────────────────────────
        self.pantalla.blit(self.fuente_chica_bold.render("CONTROL DE TIEMPO", True, COLOR_SUBTIT), (x0 + 20, 20))

        def dibujar_reloj(y, titulo, tiempo, activo):
            color_caja  = (46, 160, 67) if activo and not self.modo_visor else (45, 48, 56)
            color_texto = (255, 255, 255) if activo and not self.modo_visor else (180, 185, 195)
            
            caja = pygame.Rect(x0 + 20, y, 210, 65)
            pygame.draw.rect(self.pantalla, color_caja, caja, border_radius=8)
            
            if self.modo_visor:
                texto_tiempo = "No disp."
            else:
                m, s = divmod(int(tiempo), 60)
                texto_tiempo = f"{m:02d}:{s:02d}"

            self.pantalla.blit(self.fuente_chica.render(titulo, True, color_texto), (caja.x + 15, caja.y + 10))
            self.pantalla.blit(self.fuente.render(texto_tiempo, True, color_texto), (caja.x + 15, caja.y + 32))

        dibujar_reloj(55, "Blancas", self.tiempos[1],  self.turno == 1)
        dibujar_reloj(135, "Negras", self.tiempos[-1], self.turno == -1)

        # ── Separador y Mejor Movimiento ───────────────────────────────────
        sep_y = 230
        pygame.draw.line(self.pantalla, (55, 60, 70), (x0 + 20, sep_y), (x0 + 230, sep_y), 2)
        
        self.pantalla.blit(self.fuente_chica_bold.render("MEJOR MOVIMIENTO", True, COLOR_SUBTIT), (x0 + 20, sep_y + 15))

        def dibujar_sugerencia(y, titulo, datos_sug, color_acento):
            caja = pygame.Rect(x0 + 20, y, 210, 60)
            pygame.draw.rect(self.pantalla, (35, 38, 45), caja, border_radius=8)
            pygame.draw.rect(self.pantalla, color_acento, pygame.Rect(caja.x, caja.y, 4, caja.height), border_radius=4)

            self.pantalla.blit(self.fuente_mini.render(titulo, True, color_acento), (caja.x + 15, caja.y + 10))

            texto_mostrar = ""
            if isinstance(datos_sug, tuple) and len(datos_sug) > 0:
                texto_mostrar = str(datos_sug[0])
            elif datos_sug is not None:
                texto_mostrar = str(datos_sug)

            # Renderizado seguro
            if texto_mostrar:
                txt_render = self.fuente.render(texto_mostrar, True, (220, 220, 220))
            elif self.resultado_final is None:
                txt_render = self.fuente_chica.render("analizando...", True, (100, 105, 115))
            else:
                txt_render = self.fuente_chica.render("", True, (0, 0, 0))

            self.pantalla.blit(txt_render, (caja.x + 15, caja.y + 28))

        sug = self.sugerencias
        dibujar_sugerencia(sep_y + 45, "BLANCAS", sug.get('blancas'), COLOR_BLANCAS)
        dibujar_sugerencia(sep_y + 115, "NEGRAS",  sug.get('negras'),  COLOR_NEGRAS)

    def _dibujar_resaltados(self):
        """Traduce el UCI ('e2e4') a coordenadas X,Y de la matriz y dibuja los rectángulos."""
        COLOR_BLANCAS = (255, 210, 80)
        COLOR_NEGRAS  = (90, 170, 255)
        
        for key, color in [('blancas', COLOR_BLANCAS), ('negras', COLOR_NEGRAS)]:
            sug = self.sugerencias.get(key)
            if isinstance(sug, tuple) and len(sug) > 1 and sug[1]:
                uci = sug[1]
                
                # Conversión de UCI a índices de matriz 0-7
                c1 = ord(uci[0]) - ord('a')
                r1 = 8 - int(uci[1])
                c2 = ord(uci[2]) - ord('a')
                r2 = 8 - int(uci[3])
                
                t = self.tamano_celda
                
                # Celda de origen
                pygame.draw.rect(self.pantalla, color, (c1*t, r1*t, t, t), 4)
                # Celda de destino
                pygame.draw.rect(self.pantalla, color, (c2*t, r2*t, t, t), 4)

    def actualizar(self, matriz, eventos_externos=None):
        if not isinstance(matriz, np.ndarray):
            raise TypeError("Se esperaba un numpy.ndarray.")
        
        if not self.corriendo: return False

        ahora = pygame.time.get_ticks()
        dt = (ahora - self.ultimo_tick) / 1000.0  
        self.ultimo_tick = ahora
        
        if self.resultado_final is None:
            self.tiempos[self.turno] += dt
            self._inferir_turno(matriz)

        eventos = eventos_externos if eventos_externos is not None else pygame.event.get()

        for evento in eventos:
            if evento.type == pygame.QUIT:
                self.corriendo = False
                # pygame.quit()
                return False

        self.pantalla.blit(self.assets['fondo'], (0, 0))
        
        filas, cols = np.nonzero(matriz)
        for f, c in zip(filas, cols):
            pieza = matriz[f, c]
            self.pantalla.blit(self.assets[pieza], (c * self.tamano_celda, f * self.tamano_celda))

        self._dibujar_resaltados()        
        self._dibujar_panel()

        if self.resultado_final:
            textos = {"1-0": "¡Ganan las Blancas!", "0-1": "¡Ganan las Negras!", "1/2-1/2": "¡Tablas!"}
            texto_mostrar = textos.get(self.resultado_final, "Fin del juego")
            
            fondo = pygame.Surface((self.ancho_tablero, 100), pygame.SRCALPHA)
            fondo.fill((0, 0, 0, 190))
            self.pantalla.blit(fondo, (0, self.ancho_tablero // 2 - 50))
            
            txt_superficie = self.fuente_grande.render(texto_mostrar, True, (255, 255, 255))
            rect = txt_superficie.get_rect(center=(self.ancho_tablero // 2, self.ancho_tablero // 2))
            self.pantalla.blit(txt_superficie, rect)
            
        pygame.display.flip()
        return True

    def set_resultado(self, resultado):
        """Activa el cartel de fin de juego."""
        self.resultado_final = resultado

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