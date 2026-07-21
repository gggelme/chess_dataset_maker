# PRUEBA CON VIDEO CON HOUGH

import cv2 as cv
import numpy as np
from collections import deque
import random
from copy import copy
from numpy.random import default_rng
import itertools
import math
import os

rng = default_rng()

# Ruta al video
script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.normpath(os.path.join(script_dir, "../../data/raw/Prueba2.mp4"))

# Abrir el video
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError(f"No se pudo abrir el video: {video_path}")

# Obtener información del video
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv.CAP_PROP_FPS)

print(f"Frames totales: {total_frames}")
print(f"FPS: {fps:.2f}")

# Crear ventanas
window_name = "Video"
sobel_window = "Sobel"
canny_window = "Canny"
canny_dilated_window = "Canny Dilatado"
hough_filtered_window = "Saddle Points"
contour_window = "Contornos"
approx_window = "Polígonos aproximados"
quadrilaterals_window = "Cuadriláteros detectados"
estimated_board_window = "Cuadrilátero con más puntos silla"
warp_window = "Warp final"
controls_window = "Controles"
hough_window = "HoughP - Líneas detectadas"
polygons_binary_window = "Máscara binaria de polígonos"
intersecciones_window = "Intersecciones - Puntos y Cuadriláteros"

cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.namedWindow(sobel_window, cv.WINDOW_NORMAL)
cv.namedWindow(canny_window, cv.WINDOW_NORMAL)
cv.namedWindow(canny_dilated_window, cv.WINDOW_NORMAL)
cv.namedWindow(hough_filtered_window, cv.WINDOW_NORMAL)
cv.namedWindow(contour_window, cv.WINDOW_NORMAL)
cv.namedWindow(approx_window, cv.WINDOW_NORMAL)
cv.namedWindow(quadrilaterals_window, cv.WINDOW_NORMAL)
cv.namedWindow(estimated_board_window, cv.WINDOW_NORMAL)
cv.namedWindow(warp_window, cv.WINDOW_NORMAL)
cv.namedWindow(controls_window, cv.WINDOW_NORMAL)
cv.namedWindow(hough_window, cv.WINDOW_NORMAL)
cv.namedWindow(polygons_binary_window, cv.WINDOW_NORMAL)
cv.namedWindow(intersecciones_window, cv.WINDOW_NORMAL)

# Variable para detectar cambios del slider
current_frame = 0

def on_trackbar(pos):
    global current_frame
    current_frame = pos

# Crear la trackbar para el frame
cv.createTrackbar(
    "Frame",
    window_name,
    0,
    total_frames - 1,
    on_trackbar
)

# Variable para el offset
offset_pixels = 0

def on_offset_trackbar(pos):
    global offset_pixels
    offset_pixels = pos

# Crear trackbar para el offset (0-50 píxeles)
cv.createTrackbar(
    "Offset",
    controls_window,
    0,
    50,
    on_offset_trackbar
)

# Variables para HoughP
HOUGH_DISTANCE_RESOLUTION = 1  # Resolución de distancia en píxeles
HOUGH_ANGLE_RESOLUTION = np.pi / 180  # Resolución angular en radianes
HOUGH_THRESHOLD = 50  # Umbral de acumulador
HOUGH_MIN_LINE_LENGTH = 50  # Longitud mínima de línea
HOUGH_MAX_LINE_GAP = 10  # Gap máximo entre segmentos

# Porcentaje de tolerancia para reciclaje (10%)
RECYCLE_THRESHOLD_PERCENT = 10

def on_hough_threshold_trackbar(pos):
    global HOUGH_THRESHOLD
    HOUGH_THRESHOLD = max(1, pos)

cv.createTrackbar(
    "Umbral Hough",
    controls_window,
    50,
    200,
    on_hough_threshold_trackbar
)

def on_hough_min_length_trackbar(pos):
    global HOUGH_MIN_LINE_LENGTH
    HOUGH_MIN_LINE_LENGTH = max(5, pos)

cv.createTrackbar(
    "Longitud mínima",
    controls_window,
    50,
    200,
    on_hough_min_length_trackbar
)

def on_hough_max_gap_trackbar(pos):
    global HOUGH_MAX_LINE_GAP
    HOUGH_MAX_LINE_GAP = max(1, pos)

cv.createTrackbar(
    "Gap máximo",
    controls_window,
    10,
    100,
    on_hough_max_gap_trackbar
)

def on_recycle_threshold_trackbar(pos):
    global RECYCLE_THRESHOLD_PERCENT
    RECYCLE_THRESHOLD_PERCENT = max(1, pos)

cv.createTrackbar(
    "Tolerancia Reciclaje %",
    controls_window,
    1,
    100,
    on_recycle_threshold_trackbar
)

last_frame = -1

# Parámetros
TOLERANCE = 5
MIN_AREA = 10
MIN_PUNTOS_SILLA = 50

# Variables para almacenar el último cuadrilátero válido
ultimo_cuadrilatero_valido = None
ultimo_warp_valido = None
ultimo_frame_warp = None
ultimos_puntos_ordenados = None
ultimo_num_puntos = 0
ultimo_area = 0

# Flag para saber si ya se ejecutó Hough
hough_ejecutado = False

def get_vertices_as_points(polygon):
    vertices = []
    for point in polygon:
        vertices.append((int(point[0][0]), int(point[0][1])))
    return vertices

def apply_warp(frame, src_points, dst_size=(800, 800)):
    dst_points = np.array([
        [0, 0],
        [dst_size[0]-1, 0],
        [dst_size[0]-1, dst_size[1]-1],
        [0, dst_size[1]-1]
    ], dtype=np.float32)
    
    src_points = np.array(src_points, dtype=np.float32)
    M = cv.getPerspectiveTransform(src_points, dst_points)
    warped = cv.warpPerspective(frame, M, dst_size)
    return warped

def aplicar_offset_a_puntos(puntos, offset, centro):
    if offset == 0:
        return puntos
    
    puntos_con_offset = []
    for punto in puntos:
        vector = centro - punto
        norm = np.linalg.norm(vector)
        if norm > 0:
            nuevo_punto = punto + (vector / norm) * offset
        else:
            nuevo_punto = punto
        puntos_con_offset.append(nuevo_punto)
    
    return np.array(puntos_con_offset, dtype=np.float32)

# ========== FUNCIONES PARA HOUGH PROBABILÍSTICO ==========

def obtener_rectas_hough(imagen_binaria, threshold, min_line_length, max_line_gap):
    """
    Obtiene líneas usando Hough Probabilístico
    Retorna una lista de líneas en formato polar (rho, theta)
    """
    if imagen_binaria is None:
        return []
    
    # Aplicar HoughP
    lines = cv.HoughLinesP(
        imagen_binaria,
        HOUGH_DISTANCE_RESOLUTION,
        HOUGH_ANGLE_RESOLUTION,
        threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    if lines is None:
        return []
    
    # Convertir a formato polar
    lineas_polar = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calcular parámetros polares (rho, theta)
        dx = x2 - x1
        dy = y2 - y1
        
        # Calcular theta (ángulo de la normal)
        if abs(dx) < 1e-6:
            theta = np.pi / 2 if dy > 0 else -np.pi / 2
        else:
            theta = np.arctan2(dx, -dy)
        
        # Calcular rho
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        
        # Calcular orientación
        theta_normalized = theta % np.pi
        es_horizontal = theta_normalized < np.pi/4 or theta_normalized > 3*np.pi/4
        
        lineas_polar.append({
            'rho': rho,
            'theta': theta,
            'puntos': [(x1, y1), (x2, y2)],
            'origen': 'horizontal' if es_horizontal else 'vertical'
        })
    
    return lineas_polar

def agrupar_lineas_por_orientacion(lineas):
    """
    Agrupa las líneas por orientación (horizontal/vertical)
    """
    horizontales = []
    verticales = []
    
    for linea in lineas:
        if linea['origen'] == 'horizontal':
            horizontales.append(linea)
        else:
            verticales.append(linea)
    
    return horizontales, verticales

def fusionar_lineas_cercanas(lineas, umbral_rho=20, umbral_theta=0.1):
    """
    Fusiona líneas que son cercanas entre sí
    """
    if len(lineas) < 2:
        return lineas
    
    fusionadas = []
    usadas = [False] * len(lineas)
    
    for i in range(len(lineas)):
        if usadas[i]:
            continue
        
        grupo = [i]
        usadas[i] = True
        
        for j in range(i + 1, len(lineas)):
            if usadas[j]:
                continue
            
            rho1, theta1 = lineas[i]['rho'], lineas[i]['theta']
            rho2, theta2 = lineas[j]['rho'], lineas[j]['theta']
            
            # Normalizar theta
            theta1_norm = theta1 % np.pi
            theta2_norm = theta2 % np.pi
            
            diff_rho = abs(rho1 - rho2)
            diff_theta = abs(theta1_norm - theta2_norm)
            diff_theta = min(diff_theta, np.pi - diff_theta)
            
            if diff_rho < umbral_rho and diff_theta < umbral_theta:
                grupo.append(j)
                usadas[j] = True
        
        # Promediar las líneas del grupo
        if len(grupo) > 0:
            rho_prom = np.mean([lineas[idx]['rho'] for idx in grupo])
            theta_prom = np.mean([lineas[idx]['theta'] for idx in grupo])
            
            # Usar los puntos de la línea con más inliers (si existiera)
            # O simplemente usar el primero
            fusionadas.append({
                'rho': rho_prom,
                'theta': theta_prom,
                'puntos': lineas[grupo[0]]['puntos'],
                'origen': lineas[grupo[0]]['origen']
            })
    
    return fusionadas

def recta_polar_a_puntos(recta_polar, ancho, alto):
    """
    Convierte una recta en representación polar (rho, theta) a dos puntos
    para dibujarla en una imagen de dimensiones (ancho, alto)
    """
    rho, theta = recta_polar
    a = np.cos(theta)
    b = np.sin(theta)
    
    # Calcular puntos extremos
    if abs(a) > 0.01:  # No es vertical
        x1 = 0
        y1 = int((rho - x1 * a) / b) if abs(b) > 1e-6 else 0
        x2 = ancho
        y2 = int((rho - x2 * a) / b) if abs(b) > 1e-6 else 0
    else:  # Línea vertical
        y1 = 0
        x1 = int(rho / a) if abs(a) > 1e-6 else 0
        y2 = alto
        x2 = int(rho / a) if abs(a) > 1e-6 else 0
    
    # Asegurar que los puntos estén dentro de la imagen
    x1 = max(0, min(ancho, x1))
    x2 = max(0, min(ancho, x2))
    y1 = max(0, min(alto, y1))
    y2 = max(0, min(alto, y2))
    
    return (x1, y1), (x2, y2)

def calcular_interseccion_polar(recta1, recta2):
    """
    Calcula la intersección de dos rectas en representación polar (rho, theta)
    Retorna el punto (x, y) o None si son paralelas
    """
    rho1, theta1 = recta1
    rho2, theta2 = recta2
    
    # Si las rectas son paralelas (theta iguales o diferencia de 180 grados)
    if abs(theta1 - theta2) < 1e-6 or abs(abs(theta1 - theta2) - np.pi) < 1e-6:
        return None
    
    # Resolver sistema de ecuaciones:
    # x*cos(theta1) + y*sin(theta1) = rho1
    # x*cos(theta2) + y*sin(theta2) = rho2
    
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    
    try:
        x, y = np.linalg.solve(A, b)
        return (int(x), int(y))
    except np.linalg.LinAlgError:
        return None

def formar_cuadrilateros_desde_rectas(rectas_horizontal, rectas_vertical, saddle_points, frame_shape):
    """
    Forma cuadriláteros a partir de las intersecciones de rectas horizontales y verticales
    """
    h, w = frame_shape[:2]
    cuadrilateros = []
    
    # Tomar las mejores líneas (más largas/confiables)
    # Ordenar por longitud de segmento
    def longitud_linea(linea):
        p1, p2 = linea['puntos']
        return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    
    rectas_horizontal.sort(key=longitud_linea, reverse=True)
    rectas_vertical.sort(key=longitud_linea, reverse=True)
    
    # Tomar las primeras N líneas
    max_lineas = min(8, len(rectas_horizontal))
    rectas_horizontal = rectas_horizontal[:max_lineas]
    max_lineas = min(8, len(rectas_vertical))
    rectas_vertical = rectas_vertical[:max_lineas]
    
    for combo_h in itertools.combinations(range(len(rectas_horizontal)), 2):
        for combo_v in itertools.combinations(range(len(rectas_vertical)), 2):
            h1 = rectas_horizontal[combo_h[0]]
            h2 = rectas_horizontal[combo_h[1]]
            v1 = rectas_vertical[combo_v[0]]
            v2 = rectas_vertical[combo_v[1]]
            
            # Obtener parámetros polares
            h1_recta = (h1['rho'], h1['theta'])
            h2_recta = (h2['rho'], h2['theta'])
            v1_recta = (v1['rho'], v1['theta'])
            v2_recta = (v2['rho'], v2['theta'])
            
            # Calcular intersecciones
            inter1 = calcular_interseccion_polar(h1_recta, v1_recta)
            inter2 = calcular_interseccion_polar(h1_recta, v2_recta)
            inter3 = calcular_interseccion_polar(h2_recta, v2_recta)
            inter4 = calcular_interseccion_polar(h2_recta, v1_recta)
            
            if None in [inter1, inter2, inter3, inter4]:
                continue
            
            vertices = [inter1, inter2, inter3, inter4]
            
            # Verificar que los vértices estén dentro de la imagen
            dentro = True
            for v in vertices:
                if v[0] < 0 or v[0] >= w or v[1] < 0 or v[1] >= h:
                    dentro = False
                    break
            
            if not dentro:
                continue
            
            vertices_array = np.array(vertices, dtype=np.float32)
            vertices_ordenados = ordenar_vertices_horario(vertices_array)
            
            if vertices_ordenados is None:
                continue
            
            area = cv.contourArea(vertices_ordenados.astype(np.int32))
            if area < 100:
                continue
            
            poly_contour = vertices_ordenados.astype(np.int32).reshape(-1, 1, 2)
            if not cv.isContourConvex(poly_contour):
                continue
            
            num_puntos = contar_puntos_silla_en_poligono(poly_contour, saddle_points)
            
            if num_puntos >= MIN_PUNTOS_SILLA:
                ratio = num_puntos / np.sqrt(area) if area > 0 else 0
                cuadrilateros.append({
                    'vertices': vertices_ordenados,
                    'num_puntos': num_puntos,
                    'area': area,
                    'ratio': ratio,
                    'h1': h1,
                    'h2': h2,
                    'v1': v1,
                    'v2': v2
                })
    
    cuadrilateros.sort(key=lambda x: x['ratio'], reverse=True)
    return cuadrilateros

def dibujar_intersecciones_y_cuadrilateros(frame, rectas_horizontal, rectas_vertical, saddle_points, cuadrilateros):
    """
    Dibuja las rectas, intersecciones y cuadriláteros
    """
    img = frame.copy()
    h, w = img.shape[:2]
    
    for punto in saddle_points:
        cv.circle(img, (int(punto[0]), int(punto[1])), 2, (0, 255, 255), -1)
    
    # Dibujar rectas horizontales
    for recta in rectas_horizontal:
        rho, theta = recta['rho'], recta['theta']
        p1, p2 = recta_polar_a_puntos((rho, theta), w, h)
        cv.line(img, p1, p2, (255, 150, 0), 1)
    
    # Dibujar rectas verticales
    for recta in rectas_vertical:
        rho, theta = recta['rho'], recta['theta']
        p1, p2 = recta_polar_a_puntos((rho, theta), w, h)
        cv.line(img, p1, p2, (0, 200, 100), 1)
    
    # Dibujar intersecciones
    for h_rect in rectas_horizontal:
        h_polar = (h_rect['rho'], h_rect['theta'])
        for v_rect in rectas_vertical:
            v_polar = (v_rect['rho'], v_rect['theta'])
            inter = calcular_interseccion_polar(h_polar, v_polar)
            if inter is not None:
                x, y = inter
                if 0 <= x < w and 0 <= y < h:
                    cv.circle(img, (x, y), 4, (0, 255, 0), -1)
    
    y_offset = 30
    cv.putText(img, f"Cuadrilateros validos (min {MIN_PUNTOS_SILLA} pts): {len(cuadrilateros)}", 
              (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 25
    
    colores = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)
    ]
    
    for idx, cuad in enumerate(cuadrilateros[:8]):
        vertices = cuad['vertices']
        color = colores[idx % len(colores)]
        pts = vertices.astype(np.int32)
        cv.polylines(img, [pts], True, color, 2)
        
        centro = np.mean(vertices, axis=0).astype(int)
        cv.putText(img, f"{cuad['ratio']:.4f}", (centro[0]-20, centro[1]), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        if idx < 6:
            texto = f"#{idx+1}: {cuad['num_puntos']}pts/{cuad['area']:.0f}px = {cuad['ratio']:.4f}"
            cv.putText(img, texto, (10, y_offset), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 18
    
    if len(cuadrilateros) > 0:
        mejor = cuadrilateros[0]
        mejor_vertices = mejor['vertices']
        for v in mejor_vertices:
            cv.circle(img, (int(v[0]), int(v[1])), 8, (0, 255, 255), -1)
            cv.circle(img, (int(v[0]), int(v[1])), 10, (255, 255, 255), 1)
        
        cv.putText(img, f"MEJOR: {mejor['num_puntos']}pts / {mejor['area']:.0f}px = {mejor['ratio']:.4f}", 
                  (10, y_offset + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    cv.imshow(intersecciones_window, img)
    return img

def dibujar_lineas_hough(frame, lineas_horizontal, lineas_vertical):
    """
    Dibuja las líneas encontradas por HoughP
    """
    img = frame.copy()
    h, w = img.shape[:2]
    
    colores_h = [(255, 150, 0), (255, 100, 0), (200, 80, 0), (150, 50, 0)]
    colores_v = [(0, 200, 100), (0, 150, 80), (0, 100, 60), (0, 80, 50)]
    
    # Dibujar líneas horizontales con sus segmentos originales
    for i, recta in enumerate(lineas_horizontal):
        # Dibujar el segmento original
        p1, p2 = recta['puntos']
        color = colores_h[i % len(colores_h)]
        cv.line(img, p1, p2, color, 2)
        
        # Dibujar la línea extendida
        rho, theta = recta['rho'], recta['theta']
        p1_ext, p2_ext = recta_polar_a_puntos((rho, theta), w, h)
        cv.line(img, p1_ext, p2_ext, color, 1)
        
        cv.putText(img, f"H{i+1}", (p1[0], p1[1] - 10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Dibujar líneas verticales
    for i, recta in enumerate(lineas_vertical):
        # Dibujar el segmento original
        p1, p2 = recta['puntos']
        color = colores_v[i % len(colores_v)]
        cv.line(img, p1, p2, color, 2)
        
        # Dibujar la línea extendida
        rho, theta = recta['rho'], recta['theta']
        p1_ext, p2_ext = recta_polar_a_puntos((rho, theta), w, h)
        cv.line(img, p1_ext, p2_ext, color, 1)
        
        cv.putText(img, f"V{i+1}", (p1[0], p1[1] - 10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv.putText(img, f"Lineas H: {len(lineas_horizontal)}, Lineas V: {len(lineas_vertical)}", 
              (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv.imshow(hough_window, img)
    return img

# ========== FIN DE FUNCIONES PARA HOUGH ==========

def contar_puntos_silla_en_poligono(polygon, saddle_points):
    if polygon is None or len(saddle_points) == 0:
        return 0
    
    count = 0
    for point in saddle_points:
        distance = cv.pointPolygonTest(polygon, point, True)
        if distance >= 0:
            count += 1
    
    return count

def encontrar_cuadrilatero_con_mas_puntos_silla(cuadrilateros, saddle_points):
    if len(cuadrilateros) == 0:
        return None, 0
    
    mejor_cuadrilatero = None
    max_puntos = 0
    
    for quad in cuadrilateros:
        num_puntos = contar_puntos_silla_en_poligono(quad, saddle_points)
        if num_puntos > max_puntos and num_puntos >= MIN_PUNTOS_SILLA:
            max_puntos = num_puntos
            mejor_cuadrilatero = quad
    
    return mejor_cuadrilatero, max_puntos

def ordenar_vertices_horario(vertices):
    if len(vertices) != 4:
        return None
    
    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    vertices_ordenados = vertices[sorted_indices]
    
    vertices_ordenados = vertices_ordenados[::-1]
    
    min_sum_idx = np.argmin(vertices_ordenados[:, 0] + vertices_ordenados[:, 1])
    vertices_ordenados = np.roll(vertices_ordenados, -min_sum_idx, axis=0)
    
    return vertices_ordenados

def ordenar_puntos_para_warp(puntos):
    center = np.mean(puntos, axis=0)
    angles = np.arctan2(puntos[:, 1] - center[1], puntos[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    puntos_ordenados = puntos[sorted_indices]
    
    puntos_ordenados = puntos_ordenados[::-1]
    
    min_sum_idx = np.argmin(puntos_ordenados[:, 0] + puntos_ordenados[:, 1])
    puntos_ordenados = np.roll(puntos_ordenados, -min_sum_idx, axis=0)
    
    return puntos_ordenados

def dibujar_cuadrilatero_y_warp(frame, src_points_ordenados, num_puntos, area, offset=0, es_hough=False, es_reciclado=False):
    estimated_board_img = frame.copy()
    warp_result_img = np.zeros((800, 800, 3), dtype=np.uint8)
    
    centro = np.mean(src_points_ordenados, axis=0)
    
    if offset > 0:
        puntos_con_offset = aplicar_offset_a_puntos(src_points_ordenados, offset, centro)
    else:
        puntos_con_offset = src_points_ordenados
    
    pts = puntos_con_offset.astype(np.int32)
    cv.polylines(estimated_board_img, [pts], True, (0, 255, 0), 4)
    for point in pts:
        cv.circle(estimated_board_img, tuple(point), 10, (0, 255, 255), -1)
    
    if es_reciclado:
        metodo = "RECICLADO"
        color_texto = (0, 255, 255)
    elif es_hough:
        metodo = "HOUGH"
        color_texto = (0, 255, 255)
    else:
        metodo = "Detección"
        color_texto = (255, 255, 0)
    
    cv.putText(estimated_board_img, f"Metodo: {metodo}", 
              (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, color_texto, 2)
    cv.putText(estimated_board_img, f"Puntos silla: {num_puntos}", 
              (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.putText(estimated_board_img, f"Area: {int(area)}", 
              (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.putText(estimated_board_img, f"Offset: {offset}px", 
              (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    if es_reciclado:
        cv.putText(estimated_board_img, "RECICLADO (tolerancia)", 
                  (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    warped = apply_warp(frame, puntos_con_offset)
    warp_result_img = warped
    
    if es_reciclado:
        cv.putText(warp_result_img, "RECICLADO", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv.putText(warp_result_img, f"Puntos silla: {num_puntos}", 
              (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.putText(warp_result_img, f"Area: {int(area)}", 
              (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.putText(warp_result_img, f"Offset: {offset}px", 
              (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    h, w = warp_result_img.shape[:2]
    cell_h = h // 8
    cell_w = w // 8
    for i in range(9):
        y = i * cell_h
        cv.line(warp_result_img, (0, y), (w, y), (0, 255, 255), 1)
        x = i * cell_w
        cv.line(warp_result_img, (x, 0), (x, h), (0, 255, 255), 1)
    
    return estimated_board_img, warp_result_img

def mostrar_mensaje_hough(frame, mensaje, color=(0, 0, 255)):
    img = frame.copy()
    h, w = img.shape[:2]
    cv.putText(img, mensaje, (w//2 - 150, h//2), 
              cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv.imshow(hough_window, img)
    return img

while True:
    if current_frame != last_frame:
        cap.set(cv.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if ret:
            frame_vis = frame.copy()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # ========== PASO 1: SOBEL Y CANNY ==========
            sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
            sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
            sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))

            cv.imshow(sobel_window, sobel_magnitude)

            edges = cv.Canny(sobel_magnitude, 7000, 7050, apertureSize=5)
            cv.imshow(canny_window, edges)

            # ========== PASO 2: DILATACIÓN ==========
            kernel = np.ones((3,3), np.uint8)
            edges_dilated = cv.dilate(edges, kernel, iterations=2)
            cv.imshow(canny_dilated_window, edges_dilated)

            # ========== PASO 3: BÚSQUEDA DE CUADRILÁTERO + PUNTOS SILLA ==========
            Ixx = cv.Sobel(gray, cv.CV_32F, 2, 0, ksize=3)
            Iyy = cv.Sobel(gray, cv.CV_32F, 0, 2, ksize=3)
            Ixy = cv.Sobel(gray, cv.CV_32F, 1, 1, ksize=3)

            response = -(Ixx*Iyy - Ixy*Ixy)
            response = cv.GaussianBlur(response, (15,15), 0)

            mx = cv.dilate(response, np.ones((7,7), np.uint8))
            th = 0.15 * response.max()

            pts = np.where((response == mx) & (response > th))
            points = np.column_stack((pts[1], pts[0]))
            saddle_points = [(int(p[0]), int(p[1])) for p in points]

            saddle_points_img = frame.copy()
            if len(points) > 0:
                for point in points:
                    cv.circle(saddle_points_img, (int(point[0]), int(point[1])), 4, (0, 255, 0), -1)
            cv.imshow(hough_filtered_window, saddle_points_img)

            contours, hierarchy = cv.findContours(edges_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            contour_img = frame.copy()
            cv.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            cv.imshow(contour_window, contour_img)

            # ========== CREAR IMAGEN BINARIA ==========
            polygons_binary = np.zeros_like(gray)
            all_polygons = []
            contour_approx_img = frame.copy()
            saddle_polygons = []
            
            for contour in contours:
                epsilon = 0.01 * cv.arcLength(contour, True)
                approx = cv.approxPolyDP(contour, epsilon, True)
                all_polygons.append(approx)
                cv.drawContours(contour_approx_img, [approx], -1, (255, 0, 0), 3)
                
                cv.drawContours(polygons_binary, [approx], -1, 255, 1)
                
                if len(approx) == 4:
                    contains_saddle = False
                    for point in saddle_points:
                        distance = cv.pointPolygonTest(approx, point, True)
                        if distance >= -TOLERANCE:
                            contains_saddle = True
                            break
                    if contains_saddle:
                        saddle_polygons.append(approx)

            cv.imshow(approx_window, contour_approx_img)
            cv.imshow(polygons_binary_window, polygons_binary)

            quadrilaterals_img = frame.copy()
            quadrilaterals = []
            
            for polygon in saddle_polygons:
                if len(polygon) == 4:
                    area = cv.contourArea(polygon)
                    if cv.isContourConvex(polygon) and area >= MIN_AREA:
                        quadrilaterals.append(polygon)
                        cv.drawContours(quadrilaterals_img, [polygon], -1, (0, 0, 255), 3)
                        for point in polygon:
                            cv.circle(quadrilaterals_img, tuple(point[0]), 6, (0, 255, 255), -1)
                        cv.putText(quadrilaterals_img, f"Area: {int(area)}", 
                                  (int(polygon[0][0][0]), int(polygon[0][0][1]) - 20), 
                                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv.imshow(quadrilaterals_window, quadrilaterals_img)

            # ========== PASO 4: SELECCIONAR CUADRILÁTERO ==========
            warp_result_img = np.zeros((800, 800, 3), dtype=np.uint8)
            estimated_board_img = frame.copy()
            
            offset_actual = cv.getTrackbarPos("Offset", controls_window)
            hough_threshold = cv.getTrackbarPos("Umbral Hough", controls_window)
            hough_min_length = cv.getTrackbarPos("Longitud mínima", controls_window)
            hough_max_gap = cv.getTrackbarPos("Gap máximo", controls_window)
            recycle_threshold = cv.getTrackbarPos("Tolerancia Reciclaje %", controls_window) / 100.0
            
            # Variables de estado
            encontrado_valido = False
            usando_hough = False
            usando_reciclado = False
            
            # ===== PASO 1: BUSCAR CUADRILÁTERO CON PUNTOS DE SILLA =====
            if len(quadrilaterals) > 0 and len(saddle_points) > 0:
                mejor_quad, num_puntos = encontrar_cuadrilatero_con_mas_puntos_silla(quadrilaterals, saddle_points)
                
                if mejor_quad is not None:
                    encontrado_valido = True
                    print(f"✅ CUADRILÁTERO DETECTADO: {num_puntos} puntos silla")
                    
                    # Guardar como último válido
                    ultimo_cuadrilatero_valido = mejor_quad
                    vertices = get_vertices_as_points(mejor_quad)
                    src_points = np.array(vertices, dtype=np.float32)
                    src_points_ordenados = ordenar_puntos_para_warp(src_points)
                    
                    ultimos_puntos_ordenados = src_points_ordenados
                    ultimo_frame_warp = frame.copy()
                    area = cv.contourArea(mejor_quad)
                    ultimo_num_puntos = num_puntos
                    ultimo_area = area
                    
                    estimated_board_img, warp_result_img = dibujar_cuadrilatero_y_warp(
                        frame, src_points_ordenados, num_puntos, area, offset_actual, False, False
                    )
                    
                    ultimo_warp_valido = warp_result_img.copy()
                    
                    cv.putText(estimated_board_img, "DETECCION TRADICIONAL", 
                              (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv.putText(warp_result_img, "DETECCION TRADICIONAL", 
                              (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    cv.imshow(hough_window, np.zeros_like(frame))
                    cv.imshow(intersecciones_window, np.zeros_like(frame))
            
            # ===== PASO 2: SI NO HAY CUADRILÁTERO, USAR HOUGH =====
            if not encontrado_valido:
                if polygons_binary is not None and np.sum(polygons_binary) > 100:
                    
                    # Si Hough no se ha ejecutado nunca, ejecutarlo
                    if not hough_ejecutado:
                        print(f"🔄 EJECUTANDO HOUGH PROBABILÍSTICO")
                        usando_hough = True
                        
                        # Obtener líneas con HoughP
                        lineas = obtener_rectas_hough(
                            polygons_binary, 
                            hough_threshold,
                            hough_min_length,
                            hough_max_gap
                        )
                        
                        if len(lineas) > 0:
                            # Agrupar por orientación
                            rectas_horizontal, rectas_vertical = agrupar_lineas_por_orientacion(lineas)
                            
                            # Fusionar líneas cercanas
                            rectas_horizontal = fusionar_lineas_cercanas(rectas_horizontal)
                            rectas_vertical = fusionar_lineas_cercanas(rectas_vertical)
                            
                            print(f"   Líneas horizontales: {len(rectas_horizontal)}, verticales: {len(rectas_vertical)}")
                        else:
                            rectas_horizontal = []
                            rectas_vertical = []
                        
                        dibujar_lineas_hough(frame, rectas_horizontal, rectas_vertical)
                        
                        if len(rectas_horizontal) >= 2 and len(rectas_vertical) >= 2:
                            cuadrilateros_hough = formar_cuadrilateros_desde_rectas(
                                rectas_horizontal, rectas_vertical, saddle_points, frame.shape
                            )
                        else:
                            cuadrilateros_hough = []
                        
                        dibujar_intersecciones_y_cuadrilateros(
                            frame, rectas_horizontal, rectas_vertical, saddle_points, cuadrilateros_hough
                        )
                        
                        if len(cuadrilateros_hough) > 0:
                            mejor = cuadrilateros_hough[0]
                            mejor_vertices = mejor['vertices']
                            num_puntos = mejor['num_puntos']
                            area = mejor['area']
                            
                            # Guardar como último válido
                            src_points = np.array(mejor_vertices, dtype=np.float32)
                            src_points_ordenados = ordenar_puntos_para_warp(src_points)
                            
                            ultimos_puntos_ordenados = src_points_ordenados
                            ultimo_frame_warp = frame.copy()
                            ultimo_cuadrilatero_valido = mejor_vertices
                            ultimo_num_puntos = num_puntos
                            ultimo_area = area
                            hough_ejecutado = True
                            
                            estimated_board_img, warp_result_img = dibujar_cuadrilatero_y_warp(
                                frame, src_points_ordenados, num_puntos, area, offset_actual, True, False
                            )
                            
                            ultimo_warp_valido = warp_result_img.copy()
                            
                            cv.putText(estimated_board_img, "HOUGH (1ra vez)", 
                                      (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            cv.putText(warp_result_img, "HOUGH (1ra vez)", 
                                      (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                            print(f"✅ HOUGH exitoso: {num_puntos} puntos, área={area:.0f}")
                        else:
                            # Hough no encontró nada, reciclar si existe
                            if ultimo_cuadrilatero_valido is not None:
                                usando_reciclado = True
                                print(f"♻️ RECICLANDO (Hough sin resultados)")
                                src_points_ordenados = ultimos_puntos_ordenados
                                num_puntos = ultimo_num_puntos
                                area = ultimo_area
                                
                                estimated_board_img, warp_result_img = dibujar_cuadrilatero_y_warp(
                                    frame, src_points_ordenados, num_puntos, area, offset_actual, False, True
                                )
                                mostrar_mensaje_hough(frame, "RECICLADO (sin Hough)", (0, 255, 255))
                            else:
                                mostrar_mensaje_hough(frame, "No se encontraron cuadriláteros", (0, 0, 255))
                                cv.putText(estimated_board_img, "SIN DETECCION", 
                                          (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                cv.putText(warp_result_img, "SIN DETECCION", 
                                          (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Si Hough ya se ejecutó, reciclar con tolerancia
                    else:
                        # Contar puntos dentro del último cuadrilátero
                        poly_contour = ultimo_cuadrilatero_valido.astype(np.int32).reshape(-1, 1, 2)
                        puntos_actuales = contar_puntos_silla_en_poligono(poly_contour, saddle_points)
                        
                        if ultimo_num_puntos > 0:
                            perdida = (ultimo_num_puntos - puntos_actuales) / ultimo_num_puntos
                            
                            # Si la pérdida es menor al umbral, reciclar
                            if perdida < recycle_threshold:
                                usando_reciclado = True
                                print(f"♻️ RECICLANDO: {puntos_actuales} pts (pérdida {perdida*100:.1f}% < {recycle_threshold*100:.0f}%)")
                                
                                src_points_ordenados = ultimos_puntos_ordenados
                                num_puntos = puntos_actuales
                                area = ultimo_area
                                
                                estimated_board_img, warp_result_img = dibujar_cuadrilatero_y_warp(
                                    frame, src_points_ordenados, num_puntos, area, offset_actual, False, True
                                )
                                
                                # Actualizar el número de puntos para el siguiente frame
                                ultimo_num_puntos = puntos_actuales
                                
                                cv.imshow(hough_window, np.zeros_like(frame))
                                cv.imshow(intersecciones_window, np.zeros_like(frame))
                            else:
                                # Pérdida grande, ejecutar Hough de nuevo
                                print(f"🔄 PÉRDIDA GRANDE: {puntos_actuales} pts (pérdida {perdida*100:.1f}% >= {recycle_threshold*100:.0f}%)")
                                hough_ejecutado = False  # Forzar nueva ejecución
                                
                                # Para este frame, reciclamos con pérdida grande
                                usando_reciclado = True
                                src_points_ordenados = ultimos_puntos_ordenados
                                num_puntos = puntos_actuales
                                area = ultimo_area
                                
                                estimated_board_img, warp_result_img = dibujar_cuadrilatero_y_warp(
                                    frame, src_points_ordenados, num_puntos, area, offset_actual, False, True
                                )
                                
                                cv.putText(estimated_board_img, "RECICLADO (forzando Hough)", 
                                          (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                cv.putText(warp_result_img, "RECICLADO (forzando Hough)", 
                                          (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # Sin máscara binaria, reciclar si es posible
                    if ultimo_cuadrilatero_valido is not None:
                        usando_reciclado = True
                        print(f"♻️ RECICLANDO (sin máscara binaria)")
                        src_points_ordenados = ultimos_puntos_ordenados
                        num_puntos = ultimo_num_puntos
                        area = ultimo_area
                        
                        estimated_board_img, warp_result_img = dibujar_cuadrilatero_y_warp(
                            frame, src_points_ordenados, num_puntos, area, offset_actual, False, True
                        )
                        mostrar_mensaje_hough(frame, "RECICLADO (sin máscara)", (0, 255, 255))
                    else:
                        cv.putText(estimated_board_img, "SIN DETECCION", 
                                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv.putText(warp_result_img, "SIN DETECCION", 
                                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        mostrar_mensaje_hough(frame, "SIN DETECCION", (0, 0, 255))
            
            cv.imshow(estimated_board_window, estimated_board_img)
            cv.imshow(warp_window, warp_result_img)
            cv.imshow(window_name, frame_vis)

            last_frame = current_frame

    key = cv.waitKey(20) & 0xFF

    if key == 27:
        break

cap.release()
cv.destroyAllWindows()