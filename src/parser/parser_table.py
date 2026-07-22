import cv2 as cv
import numpy as np
from collections import deque
import itertools
import os

# UTILIDADES GEOMÉTRICAS Y CV
DST_SIZE = (800, 800)
ETIQUETAS = ["TL", "TR", "BR", "BL"]

def ordenar_puntos_para_warp(puntos):
    centro = np.mean(puntos, axis=0)
    angulos = np.arctan2(puntos[:, 1] - centro[1], puntos[:, 0] - centro[0])
    ordenados = puntos[np.argsort(angulos)[::-1]]
    return np.roll(ordenados, -np.argmin(ordenados[:, 0] + ordenados[:, 1]), axis=0)

def apply_warp(frame, src_points, dst_size=DST_SIZE):
    dst_points = np.array([[0, 0], [dst_size[0] - 1, 0], [dst_size[0] - 1, dst_size[1] - 1], [0, dst_size[1] - 1]], dtype=np.float32)
    M = cv.getPerspectiveTransform(np.array(src_points, dtype=np.float32), dst_points)
    return cv.warpPerspective(frame, M, dst_size)

def detectar_puntos_silla(gray):
    Ixx, Iyy, Ixy = cv.Sobel(gray, cv.CV_32F, 2, 0, ksize=3), cv.Sobel(gray, cv.CV_32F, 0, 2, ksize=3), cv.Sobel(gray, cv.CV_32F, 1, 1, ksize=3)
    response = cv.GaussianBlur(-(Ixx * Iyy - Ixy * Ixy), (15, 15), 0)
    mx = cv.dilate(response, np.ones((7, 7), np.uint8))
    ys, xs = np.where((response == mx) & (response > 0.15 * response.max()))
    return [(int(x), int(y)) for x, y in zip(xs, ys)]

def contar_puntos_silla_en_poligono(polygon, saddle_points, tolerancia=5):
    if polygon is None or not saddle_points: return 0
    return sum(1 for p in saddle_points if cv.pointPolygonTest(polygon, p, True) >= -tolerancia)

def procesar_contornos(gray, saddle_points):
    sobel_mag = np.uint8(np.clip(np.sqrt(cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)**2 + cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)**2), 0, 255))
    canny = cv.dilate(cv.Canny(sobel_mag, 7000, 7050, apertureSize=5), np.ones((3, 3), np.uint8), iterations=2)
    contornos, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    mascara_binaria = np.zeros_like(gray)
    validos = []
    
    for c in contornos:
        approx = cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True)
        cv.drawContours(mascara_binaria, [approx], -1, 255, 1)
        
        if len(approx) == 4 and cv.isContourConvex(approx) and cv.contourArea(approx) >= 10:
            pts_silla = contar_puntos_silla_en_poligono(approx, saddle_points)
            if pts_silla >= 50:
                validos.append({"vertices": approx, "num_puntos": pts_silla})
    
    mejor_cuad = max(validos, key=lambda c: c["num_puntos"]) if validos else None
    return mejor_cuad, mascara_binaria

# BACK-UP: HOUGH
def calcular_interseccion_polar(rho1, theta1, rho2, theta2):
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    try:
        x, y = np.linalg.solve(A, b)
        return (int(x), int(y))
    except np.linalg.LinAlgError:
        return None

def fallback_hough(mascara_binaria, saddle_points, shape):
    lineas = cv.HoughLinesP(mascara_binaria, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=40)
    if lineas is None: return None

    h_lines, v_lines = [], []
    for x1, y1, x2, y2 in lineas[:, 0]:
        theta = np.arctan2(x2 - x1, -(y2 - y1)) if abs(x2 - x1) >= 1e-6 else (np.pi / 2 if y2 > y1 else -np.pi / 2)
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        
        # Clasificar en horizontales y verticales
        if (theta % np.pi) < np.pi / 4 or (theta % np.pi) > 3 * np.pi / 4:
            h_lines.append((rho, theta))
        else:
            v_lines.append((rho, theta))

    h_lines, v_lines = h_lines[:8], v_lines[:8] # Limitar combinaciones
    cuadrilateros = []
    
    for h_pair in itertools.combinations(h_lines, 2):
        for v_pair in itertools.combinations(v_lines, 2):
            pts = [
                calcular_interseccion_polar(h_pair[0][0], h_pair[0][1], v_pair[0][0], v_pair[0][1]),
                calcular_interseccion_polar(h_pair[0][0], h_pair[0][1], v_pair[1][0], v_pair[1][1]),
                calcular_interseccion_polar(h_pair[1][0], h_pair[1][1], v_pair[1][0], v_pair[1][1]),
                calcular_interseccion_polar(h_pair[1][0], h_pair[1][1], v_pair[0][0], v_pair[0][1])
            ]
            
            if None in pts: continue
            
            # Chequear límites y formar polígono
            if any(p[0] < 0 or p[0] >= shape[1] or p[1] < 0 or p[1] >= shape[0] for p in pts): continue
            
            approx = ordenar_puntos_para_warp(np.array(pts, dtype=np.float32))
            approx_int = approx.astype(np.int32).reshape(-1, 1, 2)
            
            if cv.isContourConvex(approx_int) and cv.contourArea(approx_int) >= 100:
                pts_silla = contar_puntos_silla_en_poligono(approx_int, saddle_points)
                if pts_silla >= 50:
                    cuadrilateros.append({
                        "vertices": approx_int, 
                        "num_puntos": pts_silla, 
                        "ratio": (pts_silla**2) / np.sqrt(cv.contourArea(approx_int))
                    })
                    
    return max(cuadrilateros, key=lambda c: c["ratio"]) if cuadrilateros else None

def analizar_orientacion(warp_crudo):
    h, w = warp_crudo.shape[:2]
    cell_h, cell_w = h // 8, w // 8
    
    gray = cv.cvtColor(warp_crudo, cv.COLOR_BGR2GRAY) if len(warp_crudo.shape) == 3 else warp_crudo
    e_sup = np.mean([np.mean(gray[r * cell_h:(r + 1) * cell_h, 0:w]) for r in range(2)])
    e_inf = np.mean([np.mean(gray[r * cell_h:(r + 1) * cell_h, 0:w]) for r in range(6, 8)])
    e_izq = np.mean([np.mean(gray[0:h, c * cell_w:(c + 1) * cell_w]) for c in range(2)])
    e_der = np.mean([np.mean(gray[0:h, c * cell_w:(c + 1) * cell_w]) for c in range(6, 8)])

    transponer = abs(e_izq - e_der) > abs(e_sup - e_inf)
    voltear = (e_izq if transponer else e_sup) > (e_der if transponer else e_inf)
    
    m = {"TL": 0, "TR": 1, "BR": 2, "BL": 3}
    if transponer: m = {"TL": m["TL"], "TR": m["BL"], "BR": m["BR"], "BL": m["TR"]}
    if voltear:    m = {"TL": m["BL"], "TR": m["BR"], "BR": m["TR"], "BL": m["TL"]}
    return m

def trackear_esquinas(corners_actuales, referencia):
    ref_pts = np.array([referencia[e] for e in ETIQUETAS])
    mejor_perm = min(itertools.permutations(range(4)), key=lambda perm: sum(np.linalg.norm(corners_actuales[perm[i]] - ref_pts[i]) for i in range(4)))
    return {etq: corners_actuales[mejor_perm[i]].copy() for i, etq in enumerate(ETIQUETAS)}

# CLASE PRINCIPAL DEL DETECTOR
class DetectorTablero:
    def __init__(self, vivo=False):
        self.stability_frames = 64 if vivo else 5
        self.tolerancia_reciclaje = 5.0
        
        self.ultimos_corners = None
        self.ultimo_num_puntos = 0
        self.stability_buffer = deque(maxlen=self.stability_frames)
        self.orientacion_definida = False
        self.corners_referencia = None

    def procesar_frame(self, frame, offset_px=0):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        saddle_points = detectar_puntos_silla(gray)
        corners_nuevos = None

        # 1. Reciclaje rápido
        if self.ultimos_corners is not None and self.ultimo_num_puntos > 0:
            pts_actuales = contar_puntos_silla_en_poligono(np.array(self.ultimos_corners, dtype=np.int32).reshape(-1, 1, 2), saddle_points)
            if pts_actuales >= self.ultimo_num_puntos * (1 - self.tolerancia_reciclaje / 100.0):
                corners_nuevos = self.ultimos_corners
                self.ultimo_num_puntos = pts_actuales

        # 2. Detección por contornos
        if corners_nuevos is None:
            mejor_cuad, mascara_binaria = procesar_contornos(gray, saddle_points)
            
            if mejor_cuad:
                puntos = [(int(p[0][0]), int(p[0][1])) for p in mejor_cuad["vertices"]]
                corners_nuevos = ordenar_puntos_para_warp(np.array(puntos, dtype=np.float32))
                self.ultimo_num_puntos = mejor_cuad["num_puntos"]
            else:
                # 3. BACK-UP: Detección por Hough
                mejor_hough = fallback_hough(mascara_binaria, saddle_points, frame.shape)
                if mejor_hough:
                    puntos = [(int(p[0][0]), int(p[0][1])) for p in mejor_hough["vertices"]]
                    corners_nuevos = ordenar_puntos_para_warp(np.array(puntos, dtype=np.float32))
                    self.ultimo_num_puntos = mejor_hough["num_puntos"]

        # 4. Estabilidad, Orientación y Warp
        if corners_nuevos is not None:
            self.ultimos_corners = corners_nuevos

            if not self.orientacion_definida:
                self.stability_buffer.append(corners_nuevos.copy())
                dists = [np.mean([np.linalg.norm(corners_nuevos[i] - v_ant[i]) for v_ant in self.stability_buffer]) for i in range(4)]
                es_estable = np.mean(dists) < 15.0 

                warp_crudo = apply_warp(frame, corners_nuevos)
                warp_final = warp_crudo

                if len(self.stability_buffer) == self.stability_frames and es_estable:
                    mapeo = analizar_orientacion(warp_crudo)
                    self.corners_referencia = {etq: corners_nuevos[idx].copy() for etq, idx in mapeo.items()}
                    self.orientacion_definida = True
                
                src_ordenados = corners_nuevos
            else:
                self.corners_referencia = trackear_esquinas(corners_nuevos, self.corners_referencia)
                src_ordenados = np.array([self.corners_referencia[e] for e in ETIQUETAS], dtype=np.float32)
                
                if offset_px > 0:
                    centro = np.mean(src_ordenados, axis=0)
                    src_ordenados = np.array([p + ((centro - p) / np.linalg.norm(centro - p)) * offset_px if np.linalg.norm(centro - p) > 0 else p for p in src_ordenados], dtype=np.float32)
                
                warp_final = apply_warp(frame, src_ordenados)

            return warp_final, src_ordenados

        return None, None


# BUCLE DE TESTING
if __name__ == "__main__":
    VIVO = True
    VIDEO_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw/Prueba2.mp4"))
    CAMERA_ID = 0

    cap = cv.VideoCapture(CAMERA_ID if VIVO else VIDEO_PATH)
    if not cap.isOpened():
        raise IOError("No se pudo abrir la fuente de video.")

    detector = DetectorTablero(vivo=VIVO)

    cv.namedWindow("Video Original", cv.WINDOW_NORMAL)
    cv.resizeWindow("Video Original", 450, 450)
    
    cv.namedWindow("Tablero Warpeado", cv.WINDOW_NORMAL)
    cv.resizeWindow("Tablero Warpeado", 450, 450)

    while True:
        ret, frame = cap.read()
        if not ret: break

        tablero, esquinas = detector.procesar_frame(frame, offset_px=15)

        if tablero is not None:
            cv.polylines(frame, [esquinas.astype(np.int32)], True, (0, 255, 0), 3)
            
            # Cuadrícula con mayor grosor (2px) para que no desaparezca con el redimensionado
            h, w = tablero.shape[:2]
            paso_y, paso_x = h // 8, w // 8
            
            for i in range(1, 8):
                cv.line(tablero, (0, i * paso_y), (w, i * paso_y), (0, 255, 255), 2)
                cv.line(tablero, (i * paso_x, 0), (i * paso_x, h), (0, 255, 255), 2)

            cv.imshow("Tablero Warpeado", tablero)
        else:
            cv.imshow("Tablero Warpeado", np.zeros((*DST_SIZE, 3), dtype=np.uint8))

        cv.imshow("Video Original", frame)

        if cv.waitKey(30) & 0xFF == 27: 
            break

    cap.release()
    cv.destroyAllWindows()