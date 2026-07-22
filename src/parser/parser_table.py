import cv2 as cv
import numpy as np
from collections import deque
import itertools
import os

# python src/parser/parser_table.py


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
            if pts_silla >= 35:
                validos.append({"vertices": approx, "num_puntos": pts_silla})
    
    mejor_cuad = max(validos, key=lambda c: c["num_puntos"]) if validos else None
    return mejor_cuad, mascara_binaria

def calcular_interseccion_polar(polar1, polar2):
    rho1, theta1 = polar1
    rho2, theta2 = polar2
    A = np.array([[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])
    try:
        x, y = np.linalg.solve(A, b)
        return (int(x), int(y))
    except np.linalg.LinAlgError:
        return None

def fallback_hough(mascara_binaria, saddle_points, shape):
    lineas = cv.HoughLinesP(mascara_binaria, 1, np.pi / 360, 30, minLineLength=30, maxLineGap=60)
    if lineas is None: return None

    h_lines, v_lines = [], []
    for x1, y1, x2, y2 in lineas[:, 0]:
        theta = np.arctan2(x2 - x1, -(y2 - y1)) if abs(x2 - x1) >= 1e-6 else (np.pi / 2 if y2 > y1 else -np.pi / 2)
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        longitud = np.hypot(x2 - x1, y2 - y1)
        
        if (theta % np.pi) < np.pi / 4 or (theta % np.pi) > 3 * np.pi / 4:
            h_lines.append((rho, theta, longitud))
        else:
            v_lines.append((rho, theta, longitud))

    def fusionar(lines, d_rho=20, d_theta=0.1):
        fusionadas = []
        for rho, theta, _ in sorted(lines, key=lambda x: x[2], reverse=True):
            if not any(abs(rho - fr) < d_rho and min(abs(theta - ft), np.pi - abs(theta - ft)) < d_theta for fr, ft in fusionadas):
                fusionadas.append((rho, theta))
        return fusionadas[:12]

    h_lines, v_lines = fusionar(h_lines), fusionar(v_lines)
    cuadrilateros = []
    
    for h_pair in itertools.combinations(h_lines, 2):
        for v_pair in itertools.combinations(v_lines, 2):
            pts = [
                calcular_interseccion_polar(h_pair[0], v_pair[0]), calcular_interseccion_polar(h_pair[0], v_pair[1]),
                calcular_interseccion_polar(h_pair[1], v_pair[1]), calcular_interseccion_polar(h_pair[1], v_pair[0])
            ]
            if None in pts: continue
            if any(p[0] < 0 or p[0] >= shape[1] or p[1] < 0 or p[1] >= shape[0] for p in pts): continue
            
            approx = ordenar_puntos_para_warp(np.array(pts, dtype=np.float32))
            approx_int = approx.astype(np.int32).reshape(-1, 1, 2)
            
            if cv.isContourConvex(approx_int) and cv.contourArea(approx_int) >= 100:
                pts_silla = contar_puntos_silla_en_poligono(approx_int, saddle_points)
                if pts_silla >= 35:
                    cuadrilateros.append({
                        "vertices": approx, "num_puntos": pts_silla,
                        "ratio": (pts_silla**1.5) / np.sqrt(cv.contourArea(approx_int))
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

# CLASE PRINCIPAL DEL DETECTOR ESTÁTICO
class DetectorTablero:
    def __init__(self, vivo=False, refresco_frames=150, offset=0):
        self.estado = "BUSCANDO"
        
        self.stability_frames = 1 if vivo else 1
        self.stability_buffer = deque(maxlen=self.stability_frames)
        self.radio_estabilidad_px = 15.0
        
        self.corners_fijos = None
        self.refresco_frames = refresco_frames
        self.contador_frames = 0
        
        self.offset = offset

        self.referencia_historica = None 

    def procesar_frame(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # ==========================================
        # ESTADO: Búsqueda y Orientación
        # ==========================================
        if self.estado == "BUSCANDO":
            saddle_points = detectar_puntos_silla(gray)
            corners_nuevos = None
            
            mejor_cuad, mascara_binaria = procesar_contornos(gray, saddle_points)
            if mejor_cuad:
                puntos = [(int(p[0][0]), int(p[0][1])) for p in mejor_cuad["vertices"]]
                corners_nuevos = ordenar_puntos_para_warp(np.array(puntos, dtype=np.float32))
            else:
                mejor_hough = fallback_hough(mascara_binaria, saddle_points, frame.shape)
                if mejor_hough:
                    corners_nuevos = mejor_hough["vertices"]

            if corners_nuevos is not None:
                # Estabilidad
                if len(self.stability_buffer) == 0:
                    self.stability_buffer.append(corners_nuevos.copy())
                else:
                    ancla = self.stability_buffer[0]
                    distancias = [np.linalg.norm(corners_nuevos[i] - ancla[i]) for i in range(4)]
                    if all(d <= self.radio_estabilidad_px for d in distancias):
                        self.stability_buffer.append(corners_nuevos.copy())
                    else:
                        self.stability_buffer.clear()
                        self.stability_buffer.append(corners_nuevos.copy())
                
                # Fijar tablero
                if len(self.stability_buffer) == self.stability_frames:
                    corners_estables = self.stability_buffer[-1]
                    
                    # --- LÓGICA DE ORIENTACIÓN CON MEMORIA ---
                    if self.referencia_historica is None:
                        # 1ra vez: Analiza por energía y guarda la referencia
                        warp_crudo = apply_warp(frame, corners_estables)
                        mapeo = analizar_orientacion(warp_crudo)
                        self.corners_fijos = np.array([corners_estables[mapeo[etq]] for etq in ETIQUETAS], dtype=np.float32)
                    else:
                        # Refrescos siguientes: Empareja por distancia a la última posición conocida
                        # itertools.permutations evalúa todas las formas posibles de asignar las 4 nuevas esquinas
                        # a las 4 etiquetas, y se queda con la que suma la menor distancia total (soporta rotaciones).
                        mejor_perm = min(
                            itertools.permutations(range(4)), 
                            key=lambda perm: sum(
                                np.linalg.norm(corners_estables[perm[i]] - self.referencia_historica[ETIQUETAS[i]]) 
                                for i in range(4)
                            )
                        )
                        self.corners_fijos = np.array([corners_estables[mejor_perm[i]] for i in range(4)], dtype=np.float32)
                    
                    # Actualizamos la memoria con la posición más reciente
                    self.referencia_historica = {etq: self.corners_fijos[i] for i, etq in enumerate(ETIQUETAS)}
                    
                    self.estado = "FIJADO"
                    self.contador_frames = 0
                    self.stability_buffer.clear()
                
                return apply_warp(frame, corners_nuevos), corners_nuevos
            
            self.stability_buffer.clear()
            return None, None

        # ==========================================
        # ESTADO: Proyección Estática
        # ==========================================
        elif self.estado == "FIJADO":
            self.contador_frames += 1
            
            if self.contador_frames >= self.refresco_frames:
                self.estado = "BUSCANDO"
                self.corners_fijos = None
                return None, None

            corners_actuales = self.corners_fijos.copy()
            if self.offset > 0:
                centro = np.mean(corners_actuales, axis=0)
                corners_actuales = np.array([
                    p + ((centro - p) / np.linalg.norm(centro - p)) * self.offset if np.linalg.norm(centro - p) > 0 else p 
                    for p in corners_actuales
                ], dtype=np.float32)

            warp_final = apply_warp(frame, corners_actuales)
            return warp_final, corners_actuales


# BUCLE DE TESTING
if __name__ == "__main__":
    VIVO = True
    VIDEO_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/raw/Prueba2.mp4"))
    CAMERA_ID = 1

    cap = cv.VideoCapture(CAMERA_ID if VIVO else VIDEO_PATH)
    detector = DetectorTablero(vivo=VIVO, refresco_frames=100, offset=70) # Refresca cada 5s a 30fps

    cv.namedWindow("Video Original", cv.WINDOW_NORMAL)
    cv.namedWindow("Tablero Warpeado", cv.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret: break

        tablero, esquinas = detector.procesar_frame(frame)

        if tablero is not None:
            color = (0, 255, 0) if detector.estado == "FIJADO" else (0, 255, 255)
            cv.polylines(frame, [esquinas.astype(np.int32)], True, color, 3)
            for p in esquinas:
                cv.circle(frame, tuple(p.astype(int)), 6, color, -1)
            
            h, w = tablero.shape[:2]
            paso_y, paso_x = h // 8, w // 8
            for i in range(1, 8):
                cv.line(tablero, (0, i * paso_y), (w, i * paso_y), (0, 255, 255), 2)
                cv.line(tablero, (i * paso_x, 0), (i * paso_x, h), (0, 255, 255), 2)

            cv.imshow("Tablero Warpeado", tablero)
        else:
            cv.imshow("Tablero Warpeado", np.zeros((800, 800, 3), dtype=np.uint8))

        cv.putText(frame, f"ESTADO: {detector.estado}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        if detector.estado == "FIJADO":
            cv.putText(frame, f"REFRESCO EN: {detector.refresco_frames - detector.contador_frames} frames", (10, 65), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        cv.imshow("Video Original", frame)

        if cv.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()