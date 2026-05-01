import os
import re
import cv2
import numpy as np
from skimage import io, transform
from scipy import ndimage
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN
# ============================================================
pixel_size           = 1.1364
micras_entre_indices = 40
indices_reales       = [45, 47, 49, 51, 52, 54, 56, 58, 60, 62, 64, 67, 68, 70, 71, 73]

ruta_raiz        = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\Reconstruccion_v2"
canales          = ["C00-DAPI", "C01-IBA-1", "C02-GFAP"]
escala_reduccion = 0.2
archivo_salida   = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\volumenes\volumenes.npz"

# Parámetros Máscara Tumoral
GFAP_UMBRAL_PERCENTIL = 75
GFAP_APERTURA_PX      = 3
GFAP_CIERRE_PX        = 8

# ============================================================
# FUNCIONES DE APOYO
# ============================================================
def registrar_ecc(img_ref, img_mov):
    """Alinea cortes consecutivos para corregir desplazamientos del criostato."""
    ref_8 = cv2.normalize(img_ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mov_8 = cv2.normalize(img_mov, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criterios = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
    try:
        _, warp_matrix = cv2.findTransformECC(ref_8, mov_8, warp_matrix, cv2.MOTION_EUCLIDEAN, criterios)
        h, w = img_mov.shape
        return cv2.warpAffine(img_mov, warp_matrix, (w, h), 
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, 
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.uint16)
    except:
        return img_mov

def buscar_archivo(ruta_carpeta, n):
    patron = re.compile(rf"^c0*{n}[_.\-]|^0*{n}[_.\-]", re.IGNORECASE)
    archivos = os.listdir(ruta_carpeta)
    match = [f for f in archivos if patron.match(f)]
    return os.path.join(ruta_carpeta, match[0]) if match else None

# ============================================================
# EJECUCIÓN DEL PIPELINE
# ============================================================

# 1. Dimensiones del lienzo
print("Calculando dimensiones finales...")
max_h, max_w = 0, 0
for canal in canales:
    ruta = os.path.join(ruta_raiz, canal)
    if not os.path.exists(ruta): continue
    for f in [f for f in os.listdir(ruta) if f.lower().endswith(".tif")]:
        img_t = io.imread(os.path.join(ruta, f))
        max_h = max(max_h, int(img_t.shape[0] * escala_reduccion))
        max_w = max(max_w, int(img_t.shape[1] * escala_reduccion))
max_h, max_w = max_h + 2, max_w + 2

# 2. Procesado de volúmenes
z_min, z_max = min(indices_reales), max(indices_reales)
total_z = (z_max - z_min) * micras_entre_indices + 1
volumenes = {}

for canal in canales:
    print(f"\nProcesando: {canal}")
    ruta_c = os.path.join(ruta_raiz, canal)
    cortes = {}
    
    for n in indices_reales:
        path = buscar_archivo(ruta_c, n)
        if path:
            img = io.imread(path)
            img_res = transform.rescale(img, escala_reduccion, preserve_range=True).astype(np.uint16)
            lienzo = np.zeros((max_h, max_w), dtype=np.uint16)
            lienzo[:img_res.shape[0], :img_res.shape[1]] = img_res
            cortes[(n - z_min) * micras_entre_indices] = lienzo

    # Alineamiento
    pos_z = sorted(cortes.keys())
    print(f"  Alineando {len(pos_z)} planos...")
    for i in range(1, len(pos_z)):
        cortes[pos_z[i]] = registrar_ecc(cortes[pos_z[i-1]], cortes[pos_z[i]])

    # Interpolación lineal (relleno de los 40um)
    vol_3d = np.zeros((total_z, max_h, max_w), dtype=np.uint16)
    print(f"  Interpolando volumen...")
    for i in range(len(pos_z) - 1):
        z0, z1 = pos_z[i], pos_z[i+1]
        dist = z1 - z0
        for p in range(dist + 1):
            peso = p / dist
            vol_3d[z0 + p] = (cortes[z0]*(1-peso) + cortes[z1]*peso).astype(np.uint16)
    
    volumenes[canal] = vol_3d

# 3. Máscara GFAP (Tumor sólido)
if "C02-GFAP" in volumenes:
    print("Generando máscara para resaltar tumor...")
    vol_g = volumenes["C02-GFAP"]
    umbral = np.percentile(vol_g[vol_g > 0], GFAP_UMBRAL_PERCENTIL)
    mask = (vol_g >= umbral).astype(np.uint8)
    struct = ndimage.generate_binary_structure(3, 1)
    mask = ndimage.binary_opening(mask, structure=struct, iterations=GFAP_APERTURA_PX)
    mask = ndimage.binary_closing(mask, structure=struct, iterations=GFAP_CIERRE_PX)
    volumenes["C02-GFAP_mascara"] = (vol_g * mask).astype(np.uint16)

# 4. Guardado
metadatos = {
    "pixel_size": np.float64(pixel_size),
    "escala_reduccion": np.float64(escala_reduccion),
    "z_um_por_plano": 1.0,
    "xy_um_por_pixel": pixel_size / escala_reduccion,
    "canales": np.array(canales),
    "canales_guardados": np.array(list(volumenes.keys()))
}

os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
np.savez_compressed(archivo_salida, **{f"vol_{c.replace('-','_')}": v for c, v in volumenes.items()}, **metadatos)
print(f"\n✓ Proceso finalizado con éxito.")