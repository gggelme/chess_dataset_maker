import sys
import os

import numpy as np
import cv2


class ParserTable:

    LADO_DESTINO = 800 #tamaño dl tablro n vista

    def __init__(self, ruta: str):
        self.ruta = ruta
        self._imagen_bgr = None
        self._imagen_gris = None
        self.esquinas = None       # (4,1,2) – salida cruda de approxPolyDP
        self.tablero_hsv = None    # imagen rectificada en HSV

    # ── Paso 1: detección de bordes y esquinas ──────────────────────────────

    def detect_board_corners(self):
        """Blur → Canny → cierre morfológico → contornos → polígono de 4 esquinas."""
        gris = self._get_gris()

        desenfocado = cv2.GaussianBlur(gris, (5, 5), 0)
        bordes = cv2.Canny(desenfocado, 50, 150)

        kernel = np.ones((5, 5), np.uint8)
        bordes_cerrados = cv2.morphologyEx(bordes, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(
            bordes_cerrados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

        if len(contornos) > 0:
            c_mas_grande = contornos[0]
            hull = cv2.convexHull(c_mas_grande)
            
            perimetro = cv2.arcLength(hull, True)
            # Aumentamos la tolerancia a 0.04
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
            [0,    0   ],   # superior izquierda
            [0,    lado],   # inferior izquierda
            [lado, lado],   # inferior derecha
            [lado, 0   ],   # superior derecha
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(pts_origen, pts_destino)

        imagen_hsv = cv2.cvtColor(self._get_bgr(), cv2.COLOR_BGR2HSV)
        self.tablero_hsv = cv2.warpPerspective(imagen_hsv, H, (lado, lado))
        return self.tablero_hsv

    # ── Paso 3: estandarizar orientación ────────────────────────────────────

    def standardize_orientation(self):
        """Rota el tablero para que las blancas queden arriba y las negras abajo.

        Compara la media del canal V (brillo) en el primer y último cuarto
        del tablero en sentido vertical y horizontal.  El eje con mayor
        diferencia absoluta de medias indica la orientación lateral del tablero;
        el valor de cada media determina qué lado es más claro (blancas).
        """
        if self.tablero_hsv is None:
            self.correct_perspective()

        # Canal V de HSV = brillo, suficiente para distinguir claro/oscuro
        v = self.tablero_hsv[:, :, 2].astype(np.float32)
        lado = v.shape[0]
        cuarto = lado // 4

        media_arriba = v[:cuarto, :].mean()
        media_abajo  = v[-cuarto:, :].mean()
        media_izq    = v[:, :cuarto].mean()
        media_der    = v[:, -cuarto:].mean()

        diff_vertical   = abs(media_arriba - media_abajo)
        diff_horizontal = abs(media_izq    - media_der)

        if diff_vertical >= diff_horizontal:
            # El eje relevante es vertical (arriba/abajo)
            if media_arriba > media_abajo:
                rotacion = cv2.ROTATE_180  # blancas arriba → rotar 180 para bajar
            else:
                rotacion = None            # blancas abajo → correcto, no hacer nada
        else:
            # El eje relevante es horizontal (izquierda/derecha)
            if media_izq > media_der:
                rotacion = cv2.ROTATE_90_COUNTERCLOCKWISE  # blancas a la izq → rotar Antihorario (izq baja)
            else:
                rotacion = cv2.ROTATE_90_CLOCKWISE         # blancas a la der → rotar Horario (der baja)

        if rotacion is not None:
            self.tablero_hsv = cv2.rotate(self.tablero_hsv, rotacion)

        return self.tablero_hsv

    # ── Pipeline completo ───────────────────────────────────────────────────

    def parse(self, lado: int = LADO_DESTINO):
        """Ejecuta detección + rectificación + orientación y devuelve un objeto Board."""
        sys.path.insert(0, os.path.dirname(__file__))
        from elements.board import Board

        self.detect_board_corners()
        self.correct_perspective(lado)
        self.standardize_orientation()
        return Board(nueva_partida=False, imagen_rectificada=self.tablero_hsv)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_bgr(self):
        if self._imagen_bgr is None:
            self._imagen_bgr = cv2.imread(self.ruta)
            if self._imagen_bgr is None:
                raise FileNotFoundError(f"No se pudo cargar: {self.ruta}")
        return self._imagen_bgr

    def _get_gris(self):
        if self._imagen_gris is None:
            self._imagen_gris = cv2.imread(self.ruta, cv2.IMREAD_GRAYSCALE)
            if self._imagen_gris is None:
                raise FileNotFoundError(f"No se pudo cargar: {self.ruta}")
        return self._imagen_gris


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # rutas relativas al archivo
    BASE = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(BASE, "../../data/raw/tablero_horizontal_2.jpg")

    parser = ParserTable(ruta)

    # --- Paso 1: esquinas ---
    esquinas = parser.detect_board_corners()
    print("Esquinas detectadas:")
    print(esquinas.reshape(4, 2))

    imagen_debug = cv2.cvtColor(parser._get_bgr(), cv2.COLOR_BGR2RGB)
    cv2.drawContours(imagen_debug, [esquinas], -1, (0, 255, 0), 7)

    # --- Paso 2: perspectiva ---
    tablero_hsv = parser.correct_perspective()
    tablero_rgb = cv2.cvtColor(tablero_hsv, cv2.COLOR_HSV2RGB)

    # --- Paso 3: orientación ---
    tablero_orientado_hsv = parser.standardize_orientation()
    tablero_orientado_rgb = cv2.cvtColor(tablero_orientado_hsv, cv2.COLOR_HSV2RGB)

    # --- Board ---
    sys.path.insert(0, BASE)
    from elements.board import Board

    board = Board(nueva_partida=False, imagen_rectificada=tablero_orientado_hsv)
    print("\nMatriz del tablero (8×8):")
    print(board.matriz)
    print(f"\nCelda [0][0]: {board.celdas[0][0]}")

    img_board = board.dibujar()

    # --- Visualización ---
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    axs[0].imshow(imagen_debug)
    axs[0].set_title("1. Bordes detectados")
    axs[0].axis("off")

    axs[1].imshow(tablero_rgb)
    axs[1].set_title("2. Perspectiva corregida")
    axs[1].axis("off")

    axs[2].imshow(tablero_orientado_rgb)
    axs[2].set_title("3. Orientación estandarizada")
    axs[2].axis("off")

    axs[3].imshow(img_board)
    axs[3].set_title("4. Board con grilla")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()
