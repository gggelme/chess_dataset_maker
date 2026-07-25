
import cv2
import numpy as np
import itertools
from collections import deque

# python src/parser/table_detection.py

class DeteccionTablero:
    """
    Clase para detectar y trackear el tablero de ajedrez.
    Implementa la Fase 1 (Detección por Contornos/Hough) y Fase 3 (Tracking Lucas-Kanade)
    para tolerar oclusiones (ej. manos sobre el tablero).
    """
    def __init__(self, min_area=10, min_puntos_silla=35, frames_estabilidad=30, radio_estabilidad=15):
        self.min_area = min_area
        self.min_area_hough = 100
        self.min_puntos_silla = min_puntos_silla
        self.frames_estabilidad = frames_estabilidad
        self.radio_estabilidad = radio_estabilidad
        
        # Parámetros Canny / Hough
        self.canny_thresh1 = 7000
        self.canny_thresh2 = 7050
        self.hough_thresh = 30
        self.hough_min_len = 30
        self.hough_max_gap = 60

        # Parámetros Lucas-Kanade (Fase 3)
        self.lk_win_size = (31, 31)
        self.lk_max_level = 3
        self.lk_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        self.historia_vecindad_len = 5
        
        self.reiniciar()

    def reiniciar(self):
        """Limpia el estado y vuelve a la Fase 1 (búsqueda)."""
        self.tablero_fijado = False
        self.ancla_corners = None
        self.ancla_streak = 0
        
        self.lk_prev_gray = None
        self.lk_prev_points = None
        self.corners_actuales = None
        self.vecindad_history = {i: deque(maxlen=self.historia_vecindad_len) for i in range(4)}
        self.saddle_count_history = deque(maxlen=self.historia_vecindad_len + 1)

    def procesar_frame(self, gray, offset=15, umbral_lk_inf=3, umbral_lk_sup=50):
        """
        Analiza el frame, busca o trackea el tablero y aplica el offset.
        Retorna: (esquinas_con_offset, tablero_fijado_boolean)
        """
        saddle_points = self._detectar_puntos_silla(gray)
        puntos_base = None
        
        if not self.tablero_fijado:
            # FASE 1: Búsqueda
            esquinas = self._buscar_por_contornos(gray, saddle_points)
            if esquinas is None:
                esquinas = self._buscar_por_hough(gray, saddle_points)
                
            self._actualizar_estabilidad(esquinas, gray)
            puntos_base = esquinas if esquinas is not None else self.ancla_corners
        else:
            # FASE 3: Tracking Lucas-Kanade
            self._ejecutar_lk(gray, umbral_lk_inf, umbral_lk_sup)
            
            # Red de seguridad: Caída de puntos de silla
            poly_actual = np.array(self.corners_actuales, dtype=np.int32).reshape(-1, 1, 2)
            pts_actuales = sum(1 for p in saddle_points if cv2.pointPolygonTest(poly_actual, p, True) >= -5)
            self.saddle_count_history.append(pts_actuales)
            
            if len(self.saddle_count_history) == self.saddle_count_history.maxlen:
                pts_5_atras = self.saddle_count_history[0]
                if pts_5_atras > 0 and pts_actuales <= pts_5_atras * 0.5: # 50% de caída
                    self.reiniciar()
                    return None, False
                    
            puntos_base = self.corners_actuales

        if puntos_base is not None:
            # El offset se recibe como parámetro y se aplica aquí
            puntos_offset = self._aplicar_offset(puntos_base, offset)
            return puntos_offset, self.tablero_fijado
            
        return None, self.tablero_fijado

    # =========================================================================
    # LÓGICA DE ESTABILIDAD Y OFFSET
    # =========================================================================

    def _actualizar_estabilidad(self, corners_nuevos, gray):
        if corners_nuevos is None:
            self.ancla_corners = None
            self.ancla_streak = 0
            return

        if self.ancla_corners is None:
            self.ancla_corners = corners_nuevos
            self.ancla_streak = 1
        else:
            distancias = [np.linalg.norm(np.array(corners_nuevos[i]) - np.array(self.ancla_corners[i])) for i in range(4)]
            if all(d <= self.radio_estabilidad for d in distancias):
                self.ancla_streak += 1
                if self.ancla_streak >= self.frames_estabilidad:
                    self.tablero_fijado = True
                    self._iniciar_lk(gray, self.ancla_corners)
            else:
                self.ancla_corners = corners_nuevos
                self.ancla_streak = 1

    def _aplicar_offset(self, puntos, offset):
        if offset == 0 or puntos is None:
            return puntos
        centro = np.mean(puntos, axis=0)
        salida = []
        for punto in puntos:
            vector = centro - punto
            norm = np.linalg.norm(vector)
            salida.append(punto + (vector / norm) * offset if norm > 0 else punto)
        return np.array(salida, dtype=np.float32)

    # =========================================================================
    # LÓGICA FASE 3: LUCAS-KANADE TRACKING (Para soportar oclusiones/manos)
    # =========================================================================

    def _extraer_vecindad(self, gray, punto):
        half_win = self.lk_win_size[0] // 2
        h, w = gray.shape
        x, y = int(punto[0]), int(punto[1])
        x1, y1 = max(0, x - half_win), max(0, y - half_win)
        x2, y2 = min(w, x + half_win), min(h, y + half_win)
        if x2 <= x1 or y2 <= y1:
            return None
        return gray[y1:y2, x1:x2].copy()

    def _iniciar_lk(self, gray, corners):
        self.corners_actuales = corners.copy()
        self.lk_prev_points = corners.reshape(-1, 1, 2).astype(np.float32)
        self.lk_prev_gray = gray.copy()
        
        for i in range(4):
            self.vecindad_history[i].clear()
            vecindad = self._extraer_vecindad(gray, corners[i])
            for _ in range(self.historia_vecindad_len):
                self.vecindad_history[i].append(vecindad)

    def _ejecutar_lk(self, gray_actual, umbral_inf, umbral_sup):
        nuevos_puntos = self.lk_prev_points.copy()
        
        for i in range(4):
            punto_actual = self.lk_prev_points[i][0]
            vecindad_actual = self._extraer_vecindad(gray_actual, punto_actual)
            
            vecindad_ref = None
            for v in reversed(self.vecindad_history[i]):
                if v is not None:
                    vecindad_ref = v
                    break
            
            ejecutar_flow = True
            if vecindad_actual is not None and vecindad_ref is not None and vecindad_actual.shape == vecindad_ref.shape:
                energia = float(np.mean(cv2.absdiff(vecindad_actual, vecindad_ref)))
                if not (umbral_inf <= energia <= umbral_sup):
                    ejecutar_flow = False # Oclusión detectada -> Congelar esquina
            
            self.vecindad_history[i].append(vecindad_actual)

            if ejecutar_flow:
                next_pt, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.lk_prev_gray, gray_actual, 
                    self.lk_prev_points[i].reshape(-1, 1, 2), 
                    None, winSize=self.lk_win_size, maxLevel=self.lk_max_level, criteria=self.lk_criteria
                )
                if next_pt is not None and status is not None and status[0][0] == 1:
                    nuevos_puntos[i] = next_pt[0]

        self.lk_prev_points = nuevos_puntos
        self.lk_prev_gray = gray_actual.copy()
        self.corners_actuales = np.array([p[0] for p in nuevos_puntos], dtype=np.float32)

    # =========================================================================
    # LÓGICA FASE 1: DETECCIÓN (Puntos silla, Contornos y Hough)
    # =========================================================================

    def _detectar_puntos_silla(self, gray):
        Ixx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=3)
        Iyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=3)
        Ixy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
        response = -(Ixx * Iyy - Ixy * Ixy)
        response = cv2.GaussianBlur(response, (15, 15), 0)
        mx = cv2.dilate(response, np.ones((7, 7), np.uint8))
        th = 0.15 * response.max()
        ys, xs = np.where((response == mx) & (response > th))
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    def _buscar_por_contornos(self, gray, saddle_points):
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.uint8(np.clip(np.sqrt(sobelx**2 + sobely**2), 0, 255))
        canny = cv2.Canny(sobel_mag, self.canny_thresh1, self.canny_thresh2, apertureSize=5)
        dilatado = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=2)
        
        contornos, _ = cv2.findContours(dilatado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidatos = []
        
        for c in contornos:
            epsilon = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) == 4 and cv2.isContourConvex(approx) and cv2.contourArea(approx) >= self.min_area:
                pts_silla = sum(1 for p in saddle_points if cv2.pointPolygonTest(approx, p, True) >= -5)
                if pts_silla >= self.min_puntos_silla:
                    candidatos.append((approx, pts_silla))
                    
        if not candidatos: return None
        mejor = max(candidatos, key=lambda x: x[1])[0]
        pts = np.array([(int(p[0][0]), int(p[0][1])) for p in mejor], dtype=np.float32)
        return self._ordenar_puntos(pts)

    def _buscar_por_hough(self, gray, saddle_points):
        """Fallback si los contornos fallan (HoughLinesP)"""
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.uint8(np.clip(np.sqrt(sobelx**2 + sobely**2), 0, 255))
        canny = cv2.Canny(sobel_mag, self.canny_thresh1, self.canny_thresh2, apertureSize=5)
        dilatado = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=2)
        
        contornos, _ = cv2.findContours(dilatado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mascara_binaria = np.zeros_like(dilatado)
        for c in contornos:
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            cv2.drawContours(mascara_binaria, [approx], -1, 255, 1)

        lines = cv2.HoughLinesP(mascara_binaria, 1, np.pi/360, self.hough_thresh, 
                                minLineLength=self.hough_min_len, maxLineGap=self.hough_max_gap)
        if lines is None: return None

        lineas_polar = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            theta = (np.pi / 2 if dy > 0 else -np.pi / 2) if abs(dx) < 1e-6 else np.arctan2(dx, -dy)
            rho = x1 * np.cos(theta) + y1 * np.sin(theta)
            theta_norm = theta % np.pi
            es_horizontal = theta_norm < np.pi / 4 or theta_norm > 3 * np.pi / 4
            lineas_polar.append({"rho": rho, "theta": theta, "puntos": [(x1,y1), (x2,y2)], 
                                 "ori": "h" if es_horizontal else "v"})

        rectas_h = [l for l in lineas_polar if l["ori"] == "h"]
        rectas_v = [l for l in lineas_polar if l["ori"] == "v"]

        def fusionar(lineas):
            if len(lineas) < 2: return lineas
            fusionadas = []
            usadas = [False] * len(lineas)
            for i in range(len(lineas)):
                if usadas[i]: continue
                grupo = [i]
                usadas[i] = True
                for j in range(i + 1, len(lineas)):
                    if usadas[j]: continue
                    t1, t2 = lineas[i]["theta"] % np.pi, lineas[j]["theta"] % np.pi
                    d_theta = min(abs(t1 - t2), np.pi - abs(t1 - t2))
                    if abs(lineas[i]["rho"] - lineas[j]["rho"]) < 20 and d_theta < 0.1:
                        grupo.append(j)
                        usadas[j] = True
                rho_p = np.mean([lineas[idx]["rho"] for idx in grupo])
                theta_p = np.mean([lineas[idx]["theta"] for idx in grupo])
                fusionadas.append({"rho": rho_p, "theta": theta_p, "puntos": lineas[grupo[0]]["puntos"]})
            return fusionadas

        rectas_h = sorted(fusionar(rectas_h), key=lambda l: np.hypot(l["puntos"][1][0]-l["puntos"][0][0], l["puntos"][1][1]-l["puntos"][0][1]), reverse=True)[:15]
        rectas_v = sorted(fusionar(rectas_v), key=lambda l: np.hypot(l["puntos"][1][0]-l["puntos"][0][0], l["puntos"][1][1]-l["puntos"][0][1]), reverse=True)[:15]

        def interseccion(r1, r2):
            A = np.array([[np.cos(r1["theta"]), np.sin(r1["theta"])], [np.cos(r2["theta"]), np.sin(r2["theta"])]])
            b = np.array([r1["rho"], r2["rho"]])
            try:
                x, y = np.linalg.solve(A, b)
                return (int(x), int(y))
            except np.linalg.LinAlgError:
                return None

        h, w = gray.shape
        cuadrilateros = []
        for i1, i2 in itertools.combinations(range(len(rectas_h)), 2):
            for j1, j2 in itertools.combinations(range(len(rectas_v)), 2):
                v1, v2, v3, v4 = (interseccion(rectas_h[i1], rectas_v[j1]), interseccion(rectas_h[i1], rectas_v[j2]),
                                  interseccion(rectas_h[i2], rectas_v[j2]), interseccion(rectas_h[i2], rectas_v[j1]))
                if None in (v1, v2, v3, v4): continue
                if any(v[0] < 0 or v[0] >= w or v[1] < 0 or v[1] >= h for v in [v1, v2, v3, v4]): continue
                
                vertices = self._ordenar_puntos(np.array([v1, v2, v3, v4], dtype=np.float32))
                poly = np.array(vertices, dtype=np.int32).reshape(-1, 1, 2)
                if not cv2.isContourConvex(poly): continue
                
                area = cv2.contourArea(poly)
                if area < self.min_area_hough: continue
                
                pts_silla = sum(1 for p in saddle_points if cv2.pointPolygonTest(poly, p, True) >= -5)
                if pts_silla >= self.min_puntos_silla:
                    cuadrilateros.append((vertices, pts_silla**1.5 / area**0.5))

        if not cuadrilateros: return None
        return max(cuadrilateros, key=lambda x: x[1])[0]

    @staticmethod
    def _ordenar_puntos(puntos):
        centro = np.mean(puntos, axis=0)
        angulos = np.arctan2(puntos[:, 1] - centro[1], puntos[:, 0] - centro[0])
        orden = np.argsort(angulos)[::-1]
        ordenados = puntos[orden]
        idx_min = np.argmin(ordenados[:, 0] + ordenados[:, 1])
        return np.roll(ordenados, -idx_min, axis=0)


if __name__ == "__main__":
    cap = cv2.VideoCapture(1) # Cambia a 1 si usas una cámara externa
    parser = DeteccionTablero()
    
    cv2.namedWindow("Camara Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Tablero Warpeado (Blancas abajo)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camara Original", 450, 450)
    cv2.resizeWindow("Tablero Warpeado (Blancas abajo)", 450, 450)

    LADO_DESTINO = 800

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Le pasamos el offset de forma dinámica
        esquinas, fijado = parser.procesar_frame(gray, offset=15)
        
        try:
            if esquinas is not None:
                # 3. Dibujar el polígono exterior del tablero en la cámara
                color = (0, 255, 0) if fijado else (0, 0, 255)
                cv2.polylines(frame, [esquinas.astype(np.int32)], True, color, 3)
                
                # Solo calculamos el ROI si el tablero ya se fijó temporalmente
                if fijado:
                    # 2. Calcular homografía y roi
                    pts_dst = np.array([[0, 0], [LADO_DESTINO - 1, 0], 
                                        [LADO_DESTINO - 1, LADO_DESTINO - 1], [0, LADO_DESTINO - 1]], dtype=np.float32)
                    
                    H = cv2.getPerspectiveTransform(esquinas, pts_dst)
                    roi = cv2.warpPerspective(frame, H, (LADO_DESTINO, LADO_DESTINO))
                    
                    # 4. Proyectar la grilla 8x8
                    H_inv = np.linalg.inv(H)
                    pasos = np.linspace(0, LADO_DESTINO, 9)
                    
                    for p in pasos:
                        pts_h = np.array([[[0, p], [LADO_DESTINO, p]]], dtype=np.float32)
                        pts_v = np.array([[[p, 0], [p, LADO_DESTINO]]], dtype=np.float32)
                        
                        orig_h = cv2.perspectiveTransform(pts_h, H_inv)
                        orig_v = cv2.perspectiveTransform(pts_v, H_inv)
                        
                        cv2.line(frame, tuple(map(int, orig_h[0][0])), tuple(map(int, orig_h[0][1])), (255, 0, 0), 2)
                        cv2.line(frame, tuple(map(int, orig_v[0][0])), tuple(map(int, orig_v[0][1])), (255, 0, 0), 2)

                    cv2.imshow("Tablero Warpeado (Blancas abajo)", roi)
                else:
                    cv2.imshow("Tablero Warpeado (Blancas abajo)", np.zeros((450,450,3), dtype=np.uint8))
        except ValueError:
            pass
            
        cv2.imshow("Camara Original", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
