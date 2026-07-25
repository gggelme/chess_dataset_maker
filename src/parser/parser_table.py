import cv2
import numpy as np

# python src/parser/parser_table.py

def _sort_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)          
    d = np.diff(pts, axis=1).ravel()  
    return np.array([
        pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]
    ], dtype=np.float32)

class DetectorTablero:
    LADO_DESTINO = 800

    def __init__(self, offset=0):
        self.offset = offset
        self.alfa = 0.15          
        self.zona_muerta = 3.0    
        self.reset() # Inicializamos el estado limpio

    def reset(self):
        """Borra la memoria para forzar una nueva detección."""
        self.esquinas_referencia = None
        self.esquinas_suavizadas = None
        self.H = None
        self.paciencia_oclusion = 0
        
    def update_frame(self, frame):
        self._imagen_bgr = frame.copy()
        self._imagen_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._imagen_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.esquinas = None

    def detect_board_corners(self, prev_corners=None, umbral_diff=30):
        desenfocado = cv2.GaussianBlur(self._imagen_gris, (5, 5), 0)
        bordes = cv2.Canny(desenfocado, 50, 150)
        bordes_cerrados = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contornos, _ = cv2.findContours(bordes_cerrados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contornos:
            contorno_mayor = max(contornos, key=cv2.contourArea)
            hull = cv2.convexHull(contorno_mayor)
            approx = cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True)
            
            if len(approx) == 4:
                nuevas = approx.reshape(4, 2).astype(np.float32)
                
                if prev_corners is not None:
                    nuevas_ord = _sort_corners(nuevas)
                    prev_ord = _sort_corners(prev_corners)
                    
                    max_mov = np.max(np.linalg.norm(nuevas_ord - prev_ord, axis=1))
                    variacion_area = abs(cv2.contourArea(nuevas_ord) - cv2.contourArea(prev_ord)) / cv2.contourArea(prev_ord)
                    
                    # Calculamos la varianza del interior del polígono propuesto
                    x, y, w, h = cv2.boundingRect(np.int32(nuevas_ord))
                    recorte = self._imagen_gris[y:y+h, x:x+w]
                    varianza_actual = np.var(recorte) if recorte.size > 0 else 0
                    
                    # 1. Filtro de Oclusiones (salto brusco, cambio de área o pérdida de textura)
                    if max_mov > umbral_diff or variacion_area > 0.10 or varianza_actual < 500:
                        self.paciencia_oclusion += 1
                        
                        # Si pasa más de 15 frames (aprox 0.5 seg) tapado o deformado, asumimos que se movió físicamente
                        if self.paciencia_oclusion > 15:
                            self.reset()
                            return None
                            
                        self.esquinas = prev_corners 
                        return prev_corners
                        
                    # Si llegamos acá, el movimiento del polígono es real y válido
                    self.paciencia_oclusion = 0
                        
                    # 2. Zona Muerta (elimina el micro-ruido del sensor)
                    if max_mov < self.zona_muerta:
                        self.esquinas = prev_corners
                        return prev_corners

                # 3. Filtro EMA (suaviza transiciones válidas)
                nuevas_ord = _sort_corners(nuevas)
                if self.esquinas_suavizadas is None:
                    self.esquinas_suavizadas = nuevas_ord
                else:
                    self.esquinas_suavizadas = (self.alfa * nuevas_ord) + ((1 - self.alfa) * self.esquinas_suavizadas)
                
                self.esquinas = self.esquinas_suavizadas
                return self.esquinas
            
        raise ValueError("No se encontró tablero")

    def standardize_orientation(self):
        pts_ordenados = _sort_corners(self.esquinas)
        
        if self.esquinas_referencia is None:
            # 1. Hacemos UN SOLO warp crudo para analizar la luz
            lado = 400
            dst = np.array([[0,0], [0,lado], [lado,lado], [lado,0]], dtype=np.float32)
            H = cv2.getPerspectiveTransform(pts_ordenados, dst)
            warp = cv2.warpPerspective(self._imagen_hsv[:,:,2], H, (lado, lado))
            
            # 2. Medimos el brillo del 25% de cada extremo
            b = int(lado * 0.25)
            e_sup = np.mean(warp[:b, :])
            e_inf = np.mean(warp[-b:, :])
            e_izq = np.mean(warp[:, :b])
            e_der = np.mean(warp[:, -b:])
            
            # 3. Comparamos energías para decidir la rotación exacta
            diff_filas = abs(e_sup - e_inf)
            diff_cols = abs(e_izq - e_der)
            
            if diff_cols > diff_filas:
                # Las piezas están a los lados -> rotar 90° o 270°
                mejor_roll = 1 if e_izq > e_der else 3
            else:
                # Las piezas están arriba/abajo -> rotar 180° o dejar en 0°
                mejor_roll = 2 if e_sup > e_inf else 0
                
            self.esquinas_referencia = np.roll(pts_ordenados, mejor_roll, axis=0)
            self._pts_origen_orientados = self.esquinas_referencia.copy()
        else:
            # Lógica original para tracking (frames posteriores)
            mejor_roll, mejor_dist = 0, float('inf')
            for i in range(4):
                pts_prueba = np.roll(pts_ordenados, i, axis=0)
                dist = np.linalg.norm(pts_prueba - self.esquinas_referencia)
                if dist < mejor_dist:
                    mejor_dist, mejor_roll = dist, i
            self._pts_origen_orientados = np.roll(pts_ordenados, mejor_roll, axis=0)
            self.esquinas_referencia = self._pts_origen_orientados.copy()

    def get_board_roi(self):
        """Calcula la homografía final aplicando el offset y devuelve el ROI limpio."""
        self.standardize_orientation()
        o, lado = self.offset, self.LADO_DESTINO
        
        pts_destino = np.array([
            [-o, -o], 
            [-o, lado + o], 
            [lado + o, lado + o], 
            [lado + o, -o]
        ], dtype=np.float32)
        
        self.H = cv2.getPerspectiveTransform(self._pts_origen_orientados, pts_destino)
        roi_hsv = cv2.warpPerspective(self._imagen_hsv, self.H, (lado, lado))
        
        # Ecualización del canal V para estabilizar la luz
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi_hsv[:, :, 2] = clahe.apply(roi_hsv[:, :, 2])
        
        return cv2.cvtColor(roi_hsv, cv2.COLOR_HSV2BGR)

    def get_math_grid(self):
        paso = self.LADO_DESTINO // 8
        return [int(i * paso) for i in range(9)]


if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    parser = DetectorTablero(offset=70)
    prev_corners = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        cv2.namedWindow("Camara Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Tablero Warpeado (Blancas abajo)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camara Original", 450, 450)
        cv2.resizeWindow("Tablero Warpeado (Blancas abajo)", 450, 450)

        parser.update_frame(frame)
        
        try:
            # 1. Intentar detectar o recuperar el tablero anterior
            prev_corners = parser.detect_board_corners(prev_corners)
            
            # ATENCIÓN: Si hubo reset, saltamos este frame para que no intente dibujar nada
            if prev_corners is None:
                continue
            
            # 2. Obtener el ROI recto (esto calcula parser.H internamente)
            roi = parser.get_board_roi()
            
            # 3. Dibujar el polígono exterior del tablero en la cámara
            cv2.polylines(frame, [parser.esquinas.astype(np.int32)], True, (0, 0, 255), 3)
            
            # 4. Proyectar la grilla 8x8 hacia el frame de la cámara usando H invertida
            if parser.H is not None:
                H_inv = np.linalg.inv(parser.H)
                pasos = parser.get_math_grid()
                lado = parser.LADO_DESTINO
                
                for p in pasos:
                    pts_h = np.array([[[0, p], [lado, p]]], dtype=np.float32)
                    pts_v = np.array([[[p, 0], [p, lado]]], dtype=np.float32)
                    
                    orig_h = cv2.perspectiveTransform(pts_h, H_inv)
                    orig_v = cv2.perspectiveTransform(pts_v, H_inv)
                    
                    cv2.line(frame, tuple(map(int, orig_h[0][0])), tuple(map(int, orig_h[0][1])), (255, 0, 0), 2)
                    cv2.line(frame, tuple(map(int, orig_v[0][0])), tuple(map(int, orig_v[0][1])), (255, 0, 0), 2)
            
            # Mostrar la vista plana y recortada
            cv2.imshow("Tablero Warpeado (Blancas abajo)", roi)
            
        except ValueError:
            # Si lanza ValueError ("No se encontró tablero"), limpiamos prev_corners
            prev_corners = None
            
        cv2.imshow("Camara Original", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()