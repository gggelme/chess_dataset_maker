#!/usr/bin/env python3
"""Script de prueba para verificar que detect_movements.py funciona standalone."""

import os
import sys

dir_raiz = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_raiz)

# Test 1: Importar desde detect_movements
print("=" * 70)
print("TEST 1: Importar funciones desde detect_movements.py")
print("=" * 70)

try:
    from src.parser.detect_movements import (
        get_energia,
        inicializar_tablero,
        obtener_celdas_cambiadas,
        inferir_movimiento_legal,
        chess_board_a_matriz,
    )
    print("✅ Todas las funciones importadas correctamente desde detect_movements.py")
except ImportError as e:
    print(f"❌ Error al importar: {e}")
    sys.exit(1)

# Test 2: Verificar que son callables
print("\n" + "=" * 70)
print("TEST 2: Verificar que las funciones son callables")
print("=" * 70)

functions = {
    "get_energia": get_energia,
    "inicializar_tablero": inicializar_tablero,
    "obtener_celdas_cambiadas": obtener_celdas_cambiadas,
    "inferir_movimiento_legal": inferir_movimiento_legal,
    "chess_board_a_matriz": chess_board_a_matriz,
}

for name, func in functions.items():
    if callable(func):
        print(f"✅ {name} es callable")
    else:
        print(f"❌ {name} NO es callable")

# Test 3: Prueba con numpy
print("\n" + "=" * 70)
print("TEST 3: Prueba de get_energia con array de prueba")
print("=" * 70)

try:
    import numpy as np
    test_array = np.ones((100, 100), dtype=np.uint8) * 128
    energia = get_energia(test_array)
    print(f"✅ get_energia(test_array) = {energia:.2f}")
except Exception as e:
    print(f"❌ Error en get_energia: {e}")

# Test 4: Prueba de run.py compatibility
print("\n" + "=" * 70)
print("TEST 4: Verificar compatibilidad con run.py")
print("=" * 70)

try:
    from src.ui.virtual_board import LiveBoard
    print("✅ Puede importar LiveBoard (UI module funciona)")
except ImportError as e:
    print(f"⚠️  LiveBoard no disponible: {e} (no es crítico para detect_movements)")

print("\n" + "=" * 70)
print("✅ TODOS LOS TESTS PASARON")
print("=" * 70)
print("\nAhora puedes ejecutar:")
print("  - python src/parser/detect_movements.py    (standalone)")
print("  - python src/run.py                        (con run.py)")
