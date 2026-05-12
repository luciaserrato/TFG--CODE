import numpy as np
import napari

# ============================================================
# CONFIGURACIÓN — apunta al archivo generado por normalizar.py
# ============================================================
archivo_entrada = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\volumenes\volumenes.npz"

# ============================================================
# 1. CARGAR DATOS
# ============================================================
print(f"Cargando volúmenes desde: {archivo_entrada}")
datos = np.load(archivo_entrada, allow_pickle=False)

# Recuperar metadatos
pixel_size       = float(datos["pixel_size"])
escala_reduccion = float(datos["escala_reduccion"])
canales          = list(datos["canales"])
colores          = list(datos["colores"])

# contrast_limits puede ser shape (3, 2) [uno por canal] o (2,) [único para todos]
cl_raw = datos["contrast_limits"]
if cl_raw.ndim == 2:
    contrast_limits_por_canal = [cl_raw[i].tolist() for i in range(len(canales))]
else:
    contrast_limits_por_canal = [cl_raw.tolist()] * len(canales)

print(f"Canales encontrados: {canales}")

# ============================================================
# 2. VISUALIZACIÓN EN NAPARI
# ============================================================
viewer = napari.Viewer(ndisplay=3)

# Escala física correcta para napari:
#
#   - Eje Z:  el volumen tiene 1 plano por µm  → step_z = 1 µm
#   - Eje XY: cada píxel del array reducido mide  pixel_size / escala_reduccion  µm
#             (el rescalado agrupa píxeles, aumenta el tamaño físico por píxel)
#
# napari.scale = (µm_por_voxel_Z, µm_por_voxel_Y, µm_por_voxel_X)
xy_um_por_pixel = pixel_size / escala_reduccion   # µm reales por píxel en el array reducido
z_um_por_plano  = 40.0                              # 1 plano = 40 µm (interpolación cada 40 µm)

scale_xyz = (z_um_por_plano, xy_um_por_pixel, xy_um_por_pixel)
print(f"Escala física: Z={z_um_por_plano} µm/plano  |  XY={xy_um_por_pixel:.4f} µm/píxel")

for i, (canal, color) in enumerate(zip(canales, colores)):
    clave = f"vol_{canal.replace('-', '_')}"
    if clave not in datos:
        print(f"[AVISO] No se encontró el volumen para: {canal}")
        continue

    volume_3d = datos[clave]
    cl = contrast_limits_por_canal[i]
    print(f"Añadiendo canal '{canal}' — shape: {volume_3d.shape}  contrast: {cl}")

    viewer.add_image(
        volume_3d,
        name=canal,
        scale=scale_xyz,
        blending="additive",
        colormap=color,
        contrast_limits=cl,
        opacity=0.6 if canal == "C02-GFAP" else 1.0,
    )

print("\n✓ Visualización lista.")
napari.run()
