import os
import importlib.util
import json
import pygame
import chess
import json
from virtual_board import LiveBoard
from stockfish_advisor import StockfishAdvisor

ruta_archivo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'parser', 'detect_movements.py'))
spec = importlib.util.spec_from_file_location("detect_movements", ruta_archivo)
detect_movements = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detect_movements)

# Asignas la función para usarla normal en tu código
chess_board_a_matriz = detect_movements.chess_board_a_matriz

def visualizar_partida():
    ruta_json = os.path.join(os.path.dirname(__file__), "..", "..", "data", "log", "partida.json")
    
    try:
        with open(ruta_json, "r") as f:
            historial = json.load(f).get("movimientos", [])
    except (FileNotFoundError, json.JSONDecodeError):
        print("No hay partida guardada o el archivo está corrupto.")
        return

    if not historial:
        print("El historial está vacío.")
        return

    try:
        advisor = StockfishAdvisor()
    except Exception:
        advisor = None

    # 1. Pre-calcular matrices y objetos Board
    board = chess.Board()
    # Guardamos una tupla: (matriz_visual, tablero_logico)
    estados = [(chess_board_a_matriz(board), board.copy())]
    
    for mov in historial:
        board.push(chess.Move.from_uci(mov.replace("-", "")))
        estados.append((chess_board_a_matriz(board), board.copy()))

    # 2. Iniciar visor
    tablero = LiveBoard()
    tablero.modo_visor = True  # Desactiva el reloj
    
    indice = 0
    indice_previo = -1
    corriendo = True
    
    while corriendo:
        # 3. Analizar la jugada si nos movimos en el historial
        if indice != indice_previo:
            tablero_actual = estados[indice][1]
            
            # Verificamos si en este turno terminó la partida
            if tablero_actual.is_game_over():
                tablero.set_resultado(tablero_actual.result())
                tablero.sugerencias = {'blancas': None, 'negras': None} # Limpiamos sugerencias
            else:
                tablero.set_resultado(None) # Oculta el cartel si volvemos hacia atrás
                if advisor:
                    advisor.analizar_async(tablero_actual)
                    
            indice_previo = indice

        # Traer la sugerencia calculada en el hilo secundario
        if advisor:
            tablero.sugerencias = advisor.get_sugerencias()

        eventos = pygame.event.get()
        for evento in eventos:
            if evento.type == pygame.QUIT:
                corriendo = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    corriendo = False
                elif evento.key == pygame.K_RIGHT and indice < len(estados) - 1:
                    indice += 1
                elif evento.key == pygame.K_LEFT and indice > 0:
                    indice -= 1

            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                if tablero.rect_btn_adelante.collidepoint(evento.pos) and indice < len(estados) - 1:
                    indice += 1
                elif tablero.rect_btn_atras.collidepoint(evento.pos) and indice > 0:
                    indice -= 1

        # Pasamos la matriz actual
        if not tablero.actualizar(estados[indice][0], eventos):
            corriendo = False
            
        pygame.time.wait(30) # Evita consumir demasiada CPU

    if advisor:
        advisor.cerrar()
        
    pygame.display.set_mode((800, 600))