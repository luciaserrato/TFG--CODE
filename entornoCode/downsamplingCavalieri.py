"""
CÁLCULO DE VOLUMEN TUMORAL
==========================
Rellena las listas `indices_cortes` y `areas_pixeles` con los valores
medidos en napari o ImageJ y ejecuta el script.

Las áreas deben estar en píxeles² tal como las reporta la herramienta
de medición (sin convertir).
"""

# ============================================================
# PARÁMETROS DEL EXPERIMENTO
# ============================================================
pixel_size   = 1.1364   # µm/píxel (XY)
grosor_corte = 20       # µm — grosor físico de cada sección histológica

# Número de corte de cada sección que contiene tumor,
# en el mismo orden que aparecen en el stack (ascendente).
indices_cortes = [46, 48, 49, 52, 54, 56, 58, 60, 62, 64, 67, 68, 70, 73]

# Área del tumor en cada corte (en píxeles²).
# → En napari:  Shapes layer → Polygon/Ellipse sobre el tumor → Properties
# → En ImageJ:  Analyze > Measure  (asegúrate de tener "Area" activado)
# Debe haber exactamente un valor por cada entrada en indices_cortes.
areas_pixeles = [
    0,   # corte 46
    0,   # corte 48
    0,   # corte 49
    0,   # corte 52
    0,   # corte 54
    0,   # corte 56
    0,   # corte 58
    0,   # corte 60
    0,   # corte 62
    0,   # corte 64
    0,   # corte 67
    0,   # corte 68
    0,   # corte 70
    0,   # corte 73
]


# ============================================================
# VALIDACIÓN
# ============================================================
assert len(areas_pixeles) == len(indices_cortes), (
    f"ERROR: areas_pixeles tiene {len(areas_pixeles)} valores pero "
    f"indices_cortes tiene {len(indices_cortes)}. Deben coincidir."
)

if all(a == 0 for a in areas_pixeles):
    print("⚠  Todas las áreas son 0. Rellena la lista antes de continuar.")
    exit()

n_ceros = sum(1 for a in areas_pixeles if a == 0)
if n_ceros > 0:
    print(f"⚠  Atención: {n_ceros} corte(s) con área = 0. "
          f"¿Es correcto? (puede indicar que ese corte no tiene tumor)\n")


# ============================================================
# CÁLCULO  —  Regla trapezoidal de Cavalieri
# ============================================================
#
#  Para cada par de cortes consecutivos i e i+1:
#
#    A1, A2  = áreas en µm²
#    d       = distancia Z real entre ambos cortes (µm)
#              = diferencia de índices × grosor_corte
#    V_tramo = ( (A1 + A2) / 2 ) × d          [µm³]
#
#  Volumen total = Σ V_tramo  →  convertido a mm³ (÷ 1e9)
#
volumen_um3 = 0.0
filas = []

for i in range(len(indices_cortes) - 1):
    a1_um2 = areas_pixeles[i]     * (pixel_size ** 2)
    a2_um2 = areas_pixeles[i + 1] * (pixel_size ** 2)
    d_um   = (indices_cortes[i + 1] - indices_cortes[i]) * grosor_corte
    v_um3  = ((a1_um2 + a2_um2) / 2.0) * d_um
    volumen_um3 += v_um3
    filas.append((
        f"{indices_cortes[i]}→{indices_cortes[i+1]}",
        d_um,
        a1_um2,
        a2_um2,
        v_um3 / 1e9,
    ))

volumen_mm3 = volumen_um3 / 1e9


# ============================================================
# RESULTADO
# ============================================================
print(f"\n{'Tramo':<14} {'Dist Z (µm)':>11} {'Área 1 (µm²)':>14} "
      f"{'Área 2 (µm²)':>14} {'Vol tramo (mm³)':>16}")
print("─" * 74)
for tramo, dz, a1, a2, v in filas:
    print(f"{tramo:<14} {dz:>11.1f} {a1:>14.1f} {a2:>14.1f} {v:>16.6f}")
print("─" * 74)
print(f"\n  Volumen tumoral total:  {volumen_mm3:.4f} mm³")
print(f"                          {volumen_mm3 * 1000:.2f} µL\n")
