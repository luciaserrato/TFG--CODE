import os
import numpy as np
import napari
from skimage import io, transform

# ============================================================
# CONFIGURACIÓN
# ============================================================
pixel_size          = 1.1364   
micras_entre_indices = 40       
indices_reales      = [45, 46, 47, 48, 49, 51, 52, 54, 56, 57, 58, 60, 62, 63, 64, 67, 68, 69, 70, 71, 73]

ruta_raiz           = r"C:\Users\frans\Desktop\TFG\ct2\Reconstruccion_v2"
canales             = ["C00-DAPI", "C01-IBA-1", "C02-GFAP"]
colores             = ["blue",    "green",      "red"]
escala_reduccion    = 0.2

# ============================================================
# FUNCIONES DE APOYO CORREGIDAS
# ============================================================
def normalizar_tamano(img, target_h, target_w):
    """Ajusta la imagen al lienzo target. Si es más pequeña, rellena con negro."""
    h, w = img.shape[:2]
    
    # Creamos un lienzo negro del tamaño objetivo
    
    nuevo_lienzo = np.zeros((target_h, target_w), dtype=np.uint16)
    
    # Calculamos cuánto podemos pegar (por si la imagen fuera más grande que el target)
    h_lim = min(h, target_h)
    w_lim = min(w, target_w)
    
    # Pegamos en la esquina superior izquierda (más seguro para alineación manual)
    # Si prefieres centrado, usa: off_h = (target_h - h_lim)//2
    nuevo_lienzo[:h_lim, :w_lim] = img[:h_lim, :w_lim]
    
    return nuevo_lienzo

# ============================================================
# 1. CÁLCULO DE DIMENSIONES MÁXIMAS (Búsqueda Exhaustiva)
# ============================================================
print("Calculando dimensiones del lienzo final...")
max_h, max_w = 0, 0
for canal in canales:
    ruta_canal = os.path.join(ruta_raiz, canal)
    if not os.path.exists(ruta_canal): continue
    archivos = [f for f in os.listdir(ruta_canal) if f.lower().endswith(".tif")]
    for f in archivos:
        # Cargamos solo la forma para no saturar RAM
        img_temp = io.imread(os.path.join(ruta_canal, f))
        h, w = img_temp.shape[:2]
        max_h = max(max_h, int(h * escala_reduccion))
        max_w = max(max_w, int(w * escala_reduccion))

# Añadimos un pequeño margen de seguridad
max_h += 2
max_w += 2
print(f"Lienzo final: {max_h} x {max_w}")

# ============================================================
# 2. PROCESADO E INTERPOLACIÓN
# ============================================================
z_min, z_max = min(indices_reales), max(indices_reales)
total_z_micras = (z_max - z_min) * micras_entre_indices + 1

viewer = napari.Viewer(ndisplay=3)

for canal, color in zip(canales, colores):
    ruta_canal = os.path.join(ruta_raiz, canal)
    if not os.path.exists(ruta_canal): continue

    cortes_dict = {}
    archivos_carpeta = os.listdir(ruta_canal)
    print(f"\nProcesando canal: {canal}")
    
    for n in indices_reales:
        match = [f for f in archivos_carpeta if f.lower().startswith(f"c{n}") or f.startswith(str(n))]
        if match:
            img = io.imread(os.path.join(ruta_canal, match[0]))
            # Redimensionar
            img_res = transform.rescale(img, escala_reduccion, preserve_range=True, anti_aliasing=True).astype(np.uint16)
            # Normalizar tamaño para evitar el error de broadcast
            pos_z = (n - z_min) * micras_entre_indices
            cortes_dict[pos_z] = normalizar_tamano(img_res, max_h, max_w)

    volume_3d = np.zeros((total_z_micras, max_h, max_w), dtype=np.uint16)
    posiciones_reales = sorted(cortes_dict.keys())

    print(f"Interpolando {len(posiciones_reales)} cortes...")
    for i in range(len(posiciones_reales) - 1):
        z_ini, z_fin = posiciones_reales[i], posiciones_reales[i+1]
        img_ini, img_fin = cortes_dict[z_ini], cortes_dict[z_fin]
        
        distancia = z_fin - z_ini
        for p in range(distancia + 1):
            z_actual = z_ini + p
            if z_actual >= total_z_micras: break
            
            peso_fin = p / distancia
            peso_ini = 1 - peso_fin
            
            # Operación matemática segura (mismo shape)
            volume_3d[z_actual] = (img_ini * peso_ini + img_fin * peso_fin).astype(np.uint16)

    # Visualización
    z_scale = (1.0 / pixel_size) / escala_reduccion
    viewer.add_image(
        volume_3d,
        name=canal,
        scale=(z_scale, 1, 1),
        blending="additive", 
        colormap=color,
        contrast_limits=[0, 300],
        opacity=0.6 if canal == "C02-GFAP" else 1.0
    )

print("\n✓ Proceso finalizado.")
napari.run()