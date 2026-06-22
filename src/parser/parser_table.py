import sys
import os

import numpy as np
import cv2

# python src/parser/parser_table.py


# ── Hough helpers (module-level) ─────────────────────────────────────────────

def _unificar_lineas(lineas, umbral_rho=30, umbral_theta=np.pi / 180 * 15):
    lineas_agrupadas = []
    for linea in lineas:
        for rho, theta in linea:
            grupo = None
            for g in lineas_agrupadas:
                for u_rho, u_theta in g:
                    if (abs(u_rho - rho) < umbral_rho and abs(u_theta - theta) < umbral_theta) or \
                       (abs(u_rho + rho) < umbral_rho and
                            abs(u_theta - (theta + np.pi)) % (2 * np.pi) < umbral_theta):
                        grupo = g
                        break
                if grupo:
                    break
            if grupo:
                grupo.append((rho, theta))
            else:
                lineas_agrupadas.append([(rho, theta)])

    resultado = []
    for g in lineas_agrupadas:
        g.sort(key=lambda a: a[1])
        resultado.append(g[(len(g) - 1) // 2])
    return np.array([[l] for l in resultado], dtype=np.float32)


def _segmentar_por_angulo(lineas, umbral_theta=np.pi / 180 * 10):
    grupos = []
    for linea in lineas:
        for rho, theta in linea:
            grupo = None
            for g in grupos:
                for _, u_theta in g:
                    if (abs(u_theta - theta) < umbral_theta or
                            abs(u_theta - (theta + np.pi)) % (2 * np.pi) < umbral_theta or
                            abs(u_theta - (np.pi - theta)) < umbral_theta or
                            abs(theta - (np.pi - u_theta)) < umbral_theta):
                        grupo = g
                        break
                if grupo:
                    break
            if grupo:
                grupo.append((rho, theta))
            else:
                grupos.append([(rho, theta)])

    grupos = sorted(grupos, key=len, reverse=True)
    if len(grupos) < 2:
        raise ValueError("Hough: se detectaron menos de 2 grupos angulares de líneas.")
    a = np.array([[l] for l in grupos[0]], dtype=np.float32)
    b = np.array([[l] for l in grupos[1]], dtype=np.float32)
    return a, b


def _identificar_horiz_vert(grupo_a, grupo_b):
    """Devuelve (horiz, vert) según cuál grupo tiene theta más cercano a π/2."""
    def _media_theta(g):
        return np.mean([abs(l[0][1]) for l in g])
    ta, tb = _media_theta(grupo_a), _media_theta(grupo_b)
    if abs(ta - np.pi / 2) < abs(tb - np.pi / 2):
        return grupo_a, grupo_b
    return grupo_b, grupo_a


def _extraer_limites(lineas_horiz, lineas_vert, lado):
    """Devuelve (y_pos, x_pos): listas de 9 enteros que delimitan las 8 filas/columnas."""
    y_pos = sorted({int(round(abs(l[0][0]))) for l in lineas_horiz})
    x_pos = sorted({int(round(abs(l[0][0]))) for l in lineas_vert})

    # Añadir bordes extremos si no fueron detectados
    if not y_pos or y_pos[0] > 30:
        y_pos.insert(0, 0)
    if not y_pos or y_pos[-1] < lado - 30:
        y_pos.append(lado)
    if not x_pos or x_pos[0] > 30:
        x_pos.insert(0, 0)
    if not x_pos or x_pos[-1] < lado - 30:
        x_pos.append(lado)

    if len(y_pos) != 9:
        y_pos = _interpolar_9(y_pos, lado)
    if len(x_pos) != 9:
        x_pos = _interpolar_9(x_pos, lado)

    return y_pos, x_pos


def _interpolar_9(posiciones, lado):
    inicio = posiciones[0] if posiciones else 0
    fin    = posiciones[-1] if posiciones else lado
    return [int(round(inicio + i * (fin - inicio) / 8)) for i in range(9)]


# ── Clase principal ───────────────────────────────────────────────────────────

class ParserTable:

    LADO_DESTINO = 800

    def __init__(self, fuente):
        if isinstance(fuente, str):
            self._imagen_bgr = cv2.imread(fuente)
            if self._imagen_bgr is None:
                raise FileNotFoundError(f"No se pudo cargar: {fuente}")
            self._imagen_gris = cv2.cvtColor(self._imagen_bgr, cv2.COLOR_BGR2GRAY)
            
        elif isinstance(fuente, np.ndarray):
            # Si llega una imagen en escala de grises (2 dimensiones)
            if len(fuente.shape) == 2:
                self._imagen_gris = fuente
                self._imagen_bgr = cv2.cvtColor(fuente, cv2.COLOR_GRAY2BGR)
            # Si llega una imagen a color (BGR)
            else:
                self._imagen_bgr = fuente
                self._imagen_gris = cv2.cvtColor(fuente, cv2.COLOR_BGR2GRAY)
        else:
            raise TypeError("La entrada debe ser una ruta (str) o una imagen (np.ndarray).")

        # Generar HSV en el constructor independientemente del tipo de entrada
        self._imagen_hsv = cv2.cvtColor(self._imagen_bgr, cv2.COLOR_BGR2HSV)

        self.esquinas    = None   # (4,1,2) – salida cruda de approxPolyDP
        self.tablero_hsv = None   # imagen rectificada + orientada en HSV
        self.y_pos = None         # 9 límites de filas  (Hough)
        self.x_pos = None         # 9 límites de columnas (Hough)

    # ── Paso 1: detección de bordes y esquinas ──────────────────────────────

    def detect_board_corners(self):
        """Blur → Canny → cierre morfológico → contornos → polígono de 4 esquinas."""
        desenfocado = cv2.GaussianBlur(self._imagen_gris, (5, 5), 0)
        bordes = cv2.Canny(desenfocado, 50, 150)

        kernel = np.ones((5, 5), np.uint8)
        bordes_cerrados = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(
            bordes_cerrados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

        if contornos:
            hull = cv2.convexHull(contornos[0])
            perimetro = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.04 * perimetro, True)
            if len(approx) == 4:
                self.esquinas = approx
                return approx

        raise ValueError("No se encontró un contorno de 4 esquinas en la imagen.")

    # ── Paso 2: corrección de perspectiva ───────────────────────────────────

    def correct_perspective(self, lado: int = LADO_DESTINO):
        """Transforma el tablero detectado a un cuadrado de `lado`×`lado` en HSV."""
        if self.esquinas is None:
            self.detect_board_corners()

        pts_origen = self.esquinas.reshape(4, 2).astype(np.float32)
        pts_destino = np.array([
            [0,    0   ],
            [0,    lado],
            [lado, lado],
            [lado, 0   ],
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(pts_origen, pts_destino)
        # Transformar directamente la imagen HSV generada en el constructor
        self.tablero_hsv = cv2.warpPerspective(self._imagen_hsv, H, (lado, lado))
        return self.tablero_hsv

    # ── Paso 3: estandarizar orientación ────────────────────────────────────

    def standardize_orientation(self):
        """Rota el tablero para que las blancas queden abajo y las negras arriba."""
        if self.tablero_hsv is None:
            self.correct_perspective()

        v = self.tablero_hsv[:, :, 2].astype(np.float32)
        lado = v.shape[0]
        cuarto = lado // 4

        media_arriba = v[:cuarto, :].mean()
        media_abajo  = v[-cuarto:, :].mean()
        media_izq    = v[:, :cuarto].mean()
        media_der    = v[:, -cuarto:].mean()

        diff_vert  = abs(media_arriba - media_abajo)
        diff_horiz = abs(media_izq    - media_der)

        if diff_vert >= diff_horiz:
            rotacion = None if media_abajo >= media_arriba else cv2.ROTATE_180
        else:
            rotacion = cv2.ROTATE_90_CLOCKWISE if media_der >= media_izq \
                       else cv2.ROTATE_90_COUNTERCLOCKWISE

        if rotacion is not None:
            self.tablero_hsv = cv2.rotate(self.tablero_hsv, rotacion)

        return self.tablero_hsv

    # ── Paso 4: detectar grilla con Hough ───────────────────────────────────

    def detect_grid_lines(self):
        """Aplica Hough sobre la imagen orientada → devuelve (y_pos, x_pos)."""
        if self.tablero_hsv is None:
            self.standardize_orientation()

        gray = self.tablero_hsv[:, :, 2]
        lado = gray.shape[0]

        # Preprocesado
        blur    = cv2.GaussianBlur(gray, (9, 9), 0)
        bin_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)
        v       = np.median(bin_img)
        sigma   = 0.33
        canny   = cv2.Canny(bin_img,
                            int(max(0,   (1.0 - sigma) * v)),
                            int(min(255, (1.0 + sigma) * v)),
                            apertureSize=3)

        # Hough + doble unificación
        raw = cv2.HoughLines(canny, rho=1.0, theta=np.pi / 180,
                             threshold=200, lines=np.array([]))
        if raw is None or len(raw) == 0:
            raise ValueError("Hough no detectó líneas en la imagen rectificada.")

        lineas = _unificar_lineas(raw)
        lineas = _unificar_lineas(lineas)

        # Separar y clasificar horizontales/verticales
        grupo_a, grupo_b = _segmentar_por_angulo(lineas)
        lineas_horiz, lineas_vert = _identificar_horiz_vert(grupo_a, grupo_b)

        self.y_pos, self.x_pos = _extraer_limites(lineas_horiz, lineas_vert, lado)
        return self.y_pos, self.x_pos

    # ── Pipeline completo ───────────────────────────────────────────────────

    def parse(self, lado: int = LADO_DESTINO):
        """Detección + rectificación + orientación + Hough → Board."""
        sys.path.insert(0, os.path.dirname(__file__))
        from elements.board import Board

        self.detect_board_corners()
        self.correct_perspective(lado)
        self.standardize_orientation()
        self.detect_grid_lines()
        return Board(
            nueva_partida=False,
            imagen_rectificada=self.tablero_hsv,
            y_pos=self.y_pos,
            x_pos=self.x_pos,
        )


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    BASE = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(BASE, "../../data/raw/tablero_vertical_real.jpg")

    img_gris = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    parser = ParserTable(img_gris)

    esquinas = parser.detect_board_corners()
    print("Esquinas detectadas:")
    print(esquinas.reshape(4, 2))

    parser.correct_perspective()
    parser.standardize_orientation()
    y_pos, x_pos = parser.detect_grid_lines()
    print(f"\nLímites Y: {y_pos}")
    print(f"Límites X: {x_pos}")

    tablero_rgb = cv2.cvtColor(parser.tablero_hsv, cv2.COLOR_HSV2RGB)

    sys.path.insert(0, BASE)
    from elements.board import Board

    board = Board(
        nueva_partida=False,
        imagen_rectificada=parser.tablero_hsv,
        y_pos=y_pos,
        x_pos=x_pos,
    )
    print("\nMatriz del tablero (8×8):")
    print(board.matriz)
    print(f"\nCelda [0][0]: {board.celdas[0][0]}")

    img_board = board.dibujar()

    imagen_debug = cv2.cvtColor(parser._imagen_bgr, cv2.COLOR_BGR2RGB)
    cv2.drawContours(imagen_debug, [esquinas], -1, (0, 255, 0), 7)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(imagen_debug)
    axs[0].set_title("1. Esquinas detectadas")
    axs[0].axis("off")
    axs[1].imshow(tablero_rgb)
    axs[1].set_title("2. Rectificado y orientado")
    axs[1].axis("off")
    axs[2].imshow(img_board)
    axs[2].set_title("3. Board con grilla Hough")
    axs[2].axis("off")
    plt.tight_layout()
    plt.show()
