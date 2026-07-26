import os
import pygame
import pygame_menu
from pygame_menu.baseimage import BaseImage

def iniciar_menu(funcion_detectar, funcion_visualizar):
    pygame.init()
    pantalla = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("ChessTracker")

    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    directorio_raiz = os.path.abspath(os.path.join(directorio_actual, "..", ".."))
    
    ruta_fondo = os.path.join(directorio_raiz, "assets", "menu", "fondo_menu.png")
    ruta_fuente = os.path.join(directorio_raiz, "assets", "menu", "Montserrat-Medium.ttf")
    ruta_fuente_bold = os.path.join(directorio_raiz, "assets", "menu", "Montserrat-Bold.ttf")

    imagen_fondo = BaseImage(image_path=ruta_fondo)

    efecto_hover = pygame_menu.widgets.HighlightSelection(
        border_width=2,
        margin_x=0,
        margin_y=0)

    # --- CONFIGURACIÓN DEL TEMA ---
    tema_personalizado = pygame_menu.Theme(
        background_color=imagen_fondo,
        title_bar_style=pygame_menu.widgets.MENUBAR_STYLE_NONE,
        
        widget_font=ruta_fuente, 
        widget_font_size=24,
        widget_font_color=(192, 164, 117), 
        
        widget_background_color=(50, 48, 49), 
        widget_border_width=2,
        widget_border_color=(192, 164, 117),
        
        widget_padding=(10, 25), 
        widget_margin=(0, 35),
        
        selection_color=(235, 212, 163), 
        widget_selection_effect=efecto_hover
    )

    menu = pygame_menu.Menu('', 800, 600, theme=tema_personalizado)

    # --- TÍTULO ---
    # Forzamos la fuente Bold y el color crema (#f5e9d9)
    titulo = menu.add.label('ChessTracker', font_name=ruta_fuente_bold, font_size=60, font_color=(245, 233, 217))
    
    # Eliminamos el fondo y aseguramos que no tenga borde heredado del tema
    titulo.set_background_color((0, 0, 0, 0))
    if hasattr(titulo, 'set_border'):
        titulo.set_border(0, (0, 0, 0))
        
    # translate(x, y) mueve el título hacia arriba independientemente de los botones
    titulo.translate(0, -80) 
    titulo.set_margin(0, 40) # Mantiene una separación saludable con el primer botón

    # Envoltorios
    def btn_detectar():
        funcion_detectar()
        pygame.display.set_mode((800, 600))

    def btn_visualizar():
        funcion_visualizar()
        pygame.display.set_mode((800, 600))

    # --- BOTONES ---
    menu.add.button('  DETECTAR PARTIDA  ', btn_detectar)
    menu.add.button(' VISUALIZAR PARTIDA ', btn_visualizar)
    menu.add.button('            SALIR            ', pygame_menu.events.EXIT)

    menu.mainloop(pantalla)