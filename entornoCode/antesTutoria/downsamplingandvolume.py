"""
VISUALIZACIÓN + CÁLCULO DE VOLUMEN TUMORAL
==========================================
Flujo de uso
------------
1. Ejecuta el script  →  napari se abre con los 3 canales interpolados.
2. Selecciona la capa  "ROI_tumor"  en el panel de capas.
3. Elige la herramienta Polygon (tecla P) o Ellipse (tecla E) en la barra superior.
4. En cada corte que contiene tumor:
     - Navega con el slider Z hasta ese corte.
     - Dibuja el contorno del tumor.
     - Puedes dibujar varios polígonos en el mismo corte si el tumor es discontinuo.
5. Cuando hayas terminado todos los cortes, cierra napari (× o Ctrl+Q).
6. El script extrae automáticamente las áreas, las agrupa por corte Z real
   y calcula el volumen con la regla trapezoidal de Cavalieri.

Notas
-----
- Las áreas se miden en el espacio de píxeles reducido (escala_reduccion) y
  se convierten a µm² antes del cálculo.
- Si en un corte hay varios ROIs, sus áreas se suman.
- El script imprime un resumen y guarda los resultados en 'volumen_resultado.txt'.
"""

import os
import numpy as np
import napari
from skimage import io, transform


# ============================================================
# CONFIGURACIÓN  ← edita solo esta sección
# ============================================================
pixel_size        = 1.1364   # µm/píxel (XY), antes de reducción
z_step            = 40       # µm entre cortes consecutivos del microscopio
grosor_corte      = 20       # µm — grosor físico de cada sección
ruta_raiz         = r"C:\Users\Frans\Desktop\ct2\RECONSTRUCCION_V1"
canales           = ["C00-DAPI", "C01-IBA-1", "C02-GFAP"]
colores           = ["blue",    "green",      "red"]
escala_reduccion  = 0.5
pasos_interpolacion = 4      # cortes sintéticos entre cada par real (0 = sin interpolar)

# Archivo de salida con el resumen del volumen
archivo_resultado = r"C:\Users\Frans\Desktop\ct2\volumen_resultado.txt"


# ============================================================
# INTERPOLACIÓN Z
# ============================================================
def interpolar_stack(stack, pasos):
    if pasos == 0:
        return stack
    Z, H, W = stack.shape
    Z_nuevo = (Z - 1) * (pasos + 1) + 1
    resultado = np.empty((Z_nuevo, H, W), dtype=np.float32)
    for i in range(Z - 1):
        idx = i * (pasos + 1)
        A = stack[i].astype(np.float32)
        B = stack[i + 1].astype(np.float32)
        resultado[idx] = A
        for p in range(1, pasos + 1):
            t = p / (pasos + 1)
            resultado[idx + p] = A * (1 - t) + B * t
    resultado[-1] = stack[-1].astype(np.float32)
    return resultado.astype(np.uint16)


# ============================================================
# EXTRACCIÓN DE ÁREAS DESDE CAPA DE SHAPES
# ============================================================
def extraer_areas_por_corte(shapes_layer, n_cortes_reales, pasos):
    """
    Lee todos los polígonos/elipses de la capa de shapes y devuelve
    un diccionario  {indice_corte_real: area_total_px2}.

    El índice Z en el stack interpolado se convierte al corte real
    dividiendo por (pasos + 1).

    Parámetros
    ----------
    shapes_layer    : napari.layers.Shapes
    n_cortes_reales : int   número de cortes reales cargados
    pasos           : int   pasos_interpolacion usado al construir el stack

    Devuelve
    --------
    areas : dict {int -> float}   área en píxeles² del espacio reducido
    """
    areas = {}

    if len(shapes_layer.data) == 0:
        return areas

    for shape_data, shape_type in zip(shapes_layer.data, shapes_layer.shape_type):
        # shape_data tiene forma (N_puntos, ndim)
        # La primera coordenada es Z (en el stack interpolado)
        z_interp = int(round(shape_data[:, 0].mean()))
        # Convertir a índice de corte real
        z_real = round(z_interp / (pasos + 1))
        z_real = max(0, min(z_real, n_cortes_reales - 1))

        # Calcular área del polígono con la fórmula de Shoelace (coordenadas Y, X)
        coords_yx = shape_data[:, 1:]   # descartar la dimensión Z
        y = coords_yx[:, 0]
        x = coords_yx[:, 1]

        if shape_type in ("polygon", "rectangle"):
            # Fórmula de Shoelace
            area = 0.5 * abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            )
        elif shape_type == "ellipse":
            # napari almacena la elipse como 4 puntos de control
            # (centro, radio mayor, radio menor en YX)
            # radio a = distancia del centro al punto 0
            # radio b = distancia del centro al punto 1
            centro = coords_yx.mean(axis=0)
            a = np.linalg.norm(coords_yx[0] - centro)
            b = np.linalg.norm(coords_yx[1] - centro)
            area = np.pi * a * b
        else:
            # Para tipos no reconocidos usamos Shoelace igualmente
            area = 0.5 * abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            )

        areas[z_real] = areas.get(z_real, 0.0) + area

    return areas


# ============================================================
# CÁLCULO DE VOLUMEN  (Cavalieri trapezoidal)
# ============================================================
def calcular_volumen(areas_por_corte, n_cortes_reales, px_size_real, escala, grosor):
    """
    Parámetros
    ----------
    areas_por_corte : dict {int -> float}  área en px² del espacio reducido
    n_cortes_reales : int
    px_size_real    : float  µm/px original (antes de reducción)
    escala          : float  factor de reducción aplicado
    grosor          : float  µm por sección

    Devuelve
    --------
    volumen_mm3 : float
    filas       : list de tuplas para imprimir
    """
    # Tamaño de píxel en el espacio reducido
    px_reducido = px_size_real / escala   # µm/px en el stack visualizado

    indices_con_roi = sorted(areas_por_corte.keys())
    if len(indices_con_roi) < 2:
        return None, []

    volumen_um3 = 0.0
    filas = []

    for i in range(len(indices_con_roi) - 1):
        iz1 = indices_con_roi[i]
        iz2 = indices_con_roi[i + 1]
        a1_um2 = areas_por_corte[iz1] * (px_reducido ** 2)
        a2_um2 = areas_por_corte[iz2] * (px_reducido ** 2)
        # Distancia Z real entre estos dos cortes del stack
        d_um = (iz2 - iz1) * grosor
        v_um3 = ((a1_um2 + a2_um2) / 2.0) * d_um
        volumen_um3 += v_um3
        filas.append((f"{iz1}→{iz2}", d_um, a1_um2, a2_um2, v_um3 / 1e9))

    return volumen_um3 / 1e9, filas


# ============================================================
# CARGA DE IMÁGENES
# ============================================================
max_h, max_w = 0, 0
print("Calculando dimensiones máximas...")
for canal in canales:
    ruta = os.path.join(ruta_raiz, canal)
    if not os.path.exists(ruta):
        continue
    for f in os.listdir(ruta):
        if f.endswith(".tif"):
            shape = io.imread(os.path.join(ruta, f)).shape
            max_h = max(max_h, int(shape[0] * escala_reduccion) + 2)
            max_w = max(max_w, int(shape[1] * escala_reduccion) + 2)
print(f"Canvas: {max_h} x {max_w} px")

viewer = napari.Viewer(ndisplay=3)
n_cortes_reales = 0

for canal, color in zip(canales, colores):
    ruta_canal = os.path.join(ruta_raiz, canal)
    if not os.path.exists(ruta_canal):
        print(f"  [AVISO] No se encontró: {ruta_canal}")
        continue

    archivos = sorted(f for f in os.listdir(ruta_canal) if f.endswith(".tif"))
    if not archivos:
        continue

    print(f"Procesando {canal} ({len(archivos)} cortes)...")
    stack_canal = []

    for f in archivos:
        img = io.imread(os.path.join(ruta_canal, f))
        img_res = transform.rescale(
            img, escala_reduccion, preserve_range=True, anti_aliasing=True
        ).astype(np.uint16)
        h, w = img_res.shape
        canvas = np.zeros((max_h, max_w), dtype=np.uint16)
        canvas[:min(h, max_h), :min(w, max_w)] = img_res[:min(h, max_h), :min(w, max_w)]
        stack_canal.append(canvas)

    n_cortes_reales = len(stack_canal)   # igual para todos los canales
    stack_array = np.stack(stack_canal)

    if pasos_interpolacion > 0:
        n_orig = stack_array.shape[0]
        stack_array = interpolar_stack(stack_array, pasos_interpolacion)
        print(f"  Interpolado: {n_orig} → {stack_array.shape[0]} cortes")

    z_scale = (z_step / pixel_size) * (1.0 / escala_reduccion) / (pasos_interpolacion + 1)

    viewer.add_image(
        stack_array,
        name=canal,
        scale=(z_scale, 1, 1),
        blending="additive",
        colormap=color,
        contrast_limits=[0, 1000],
    )

# ============================================================
# AÑADIR CAPA DE SHAPES PARA DIBUJAR LOS ROIs
# ============================================================
shapes_layer = viewer.add_shapes(
    name="ROI_tumor",
    shape_type="polygon",
    edge_color="yellow",
    face_color="yellow",
    opacity=0.25,
    edge_width=2,
)

print("\n✓ Napari listo.")
print("─" * 50)
print("  1. Selecciona la capa  ROI_tumor")
print("  2. Usa Polygon (P) o Ellipse (E)")
print("  3. Dibuja el tumor en cada corte")
print("  4. Cierra napari cuando termines")
print("─" * 50)

# El viewer bloquea aquí hasta que el usuario lo cierra
napari.run()


# ============================================================
# TRAS CERRAR NAPARI: EXTRAER ÁREAS Y CALCULAR VOLUMEN
# ============================================================
print("\nExtrayendo ROIs...")
areas_por_corte = extraer_areas_por_corte(
    shapes_layer, n_cortes_reales, pasos_interpolacion
)

if not areas_por_corte:
    print("⚠  No se detectaron ROIs. Asegúrate de dibujar con la capa ROI_tumor activa.")
else:
    print(f"  Cortes con ROI: {sorted(areas_por_corte.keys())}")

    volumen_mm3, filas = calcular_volumen(
        areas_por_corte, n_cortes_reales,
        pixel_size, escala_reduccion, grosor_corte
    )

    if volumen_mm3 is None:
        print("⚠  Se necesitan al menos 2 cortes con ROI para calcular el volumen.")
    else:
        # --- Imprimir tabla ---
        cabecera = (f"\n{'Cortes':<12} {'Dist Z (µm)':>11} {'Área 1 (µm²)':>14} "
                    f"{'Área 2 (µm²)':>14} {'Vol tramo (mm³)':>16}")
        separador = "─" * 72
        print(cabecera)
        print(separador)
        lineas_tabla = []
        for tramo, dz, a1, a2, v in filas:
            linea = (f"{tramo:<12} {dz:>11.1f} {a1:>14.1f} "
                     f"{a2:>14.1f} {v:>16.6f}")
            print(linea)
            lineas_tabla.append(linea)
        print(separador)
        print(f"\n  Volumen tumoral total:  {volumen_mm3:.4f} mm³")
        print(f"                          {volumen_mm3 * 1000:.2f} µL\n")

        # --- Guardar resultado ---
        with open(archivo_resultado, "w", encoding="utf-8") as fout:
            fout.write("VOLUMEN TUMORAL\n")
            fout.write("=" * 72 + "\n")
            fout.write(f"pixel_size     = {pixel_size} µm/px\n")
            fout.write(f"grosor_corte   = {grosor_corte} µm\n")
            fout.write(f"escala_reduccion = {escala_reduccion}\n\n")
            fout.write(cabecera + "\n")
            fout.write(separador + "\n")
            for linea in lineas_tabla:
                fout.write(linea + "\n")
            fout.write(separador + "\n")
            fout.write(f"\nVolumen tumoral total:  {volumen_mm3:.4f} mm³\n")
            fout.write(f"                        {volumen_mm3 * 1000:.2f} µL\n")

        print(f"✓ Resultado guardado en:\n  {archivo_resultado}")