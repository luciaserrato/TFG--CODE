import os
import numpy as np
from skimage import io, transform
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter

# ============================================================
# CONFIGURACIÓN
# ============================================================
pixel_size           = 1.1364   # µm/píxel en XY
micras_entre_indices = 40       # µm entre cortes reales consecutivos (distancia real en Z)
indices_reales       = [45, 46, 47, 48, 49, 51, 52, 54, 56, 57, 58, 60, 62, 63, 64, 67, 68, 69, 70, 71, 73]

ruta_raiz        = r"C:\Users\JPDominguez-OfficeMR\Downloads\wetransfer_reconstruccion_v2-zip_2026-04-09_1629\Reconstruccion_v2"
canales          = ["C00-DAPI", "C01-IBA-1", "C02-GFAP"]
escala_reduccion = 0.2

# Suavizado gaussiano en Z (sigma en planos interpolados) para reducir banding.
# Aumentar para más suavidad, bajar (o poner 0) para conservar nitidez de los cortes reales.
sigma_z = 1.2

# Normalización de contraste: recorte por percentil para maximizar rango dinámico.
# Si prefieres contraste fijo, pon NORMALIZAR_CONTRASTE = False.
NORMALIZAR_CONTRASTE  = True
PERCENTIL_BAJO        = 1      # % inferior que se recorta
PERCENTIL_ALTO        = 99.5   # % superior que se recorta

# Archivo de salida
archivo_salida = r"C:\Users\JPDominguez-OfficeMR\Downloads\wetransfer_reconstruccion_v2-zip_2026-04-09_1629\volumenes_opt.npz"

# ============================================================
# FUNCIONES DE APOYO
# ============================================================
def normalizar_tamano(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pega la imagen en un lienzo negro del tamaño objetivo (float32)."""
    lienzo = np.zeros((target_h, target_w), dtype=np.float32)
    h_lim = min(img.shape[0], target_h)
    w_lim = min(img.shape[1], target_w)
    lienzo[:h_lim, :w_lim] = img[:h_lim, :w_lim]
    return lienzo


def interpolar_spline_cubico(cortes_dict: dict, total_z: int) -> np.ndarray:
    """
    Interpola un volumen 3D completo usando splines cúbicos en el eje Z.

    La interpolación se hace de forma completamente vectorizada:
    se construye un array (n_cortes, H*W), se ajusta un CubicSpline
    a lo largo de Z por cada píxel (en bloque), y se evalúa en todos
    los planos Z de salida. Esto es ~100x más rápido que el bucle
    plano-a-plano con lerp y produce transiciones mucho más suaves.

    Parámetros
    ----------
    cortes_dict : {pos_z: array float32 (H, W)}
    total_z     : número total de planos Z en el volumen de salida

    Retorna
    -------
    volume : np.ndarray float32, shape (total_z, H, W)
    """
    posiciones = np.array(sorted(cortes_dict.keys()), dtype=np.float64)
    H, W = next(iter(cortes_dict.values())).shape

    # Stack de cortes reales: shape (n_cortes, H*W)
    stack = np.stack([cortes_dict[z].reshape(-1) for z in posiciones], axis=0)  # (n, H*W)

    # Ajuste del spline cúbico con condición de frontera "not-a-knot" (por defecto)
    cs = CubicSpline(posiciones, stack, axis=0, extrapolate=False)

    # Evaluar en todos los planos Z
    z_todos = np.arange(total_z, dtype=np.float64)
    interpolado = cs(z_todos)  # shape (total_z, H*W)

    # Fuera del rango de los cortes reales → rellenar con 0 (NaN → 0)
    interpolado = np.nan_to_num(interpolado, nan=0.0)

    # Clip para evitar valores negativos por las oscilaciones del spline
    np.clip(interpolado, 0, None, out=interpolado)

    return interpolado.reshape(total_z, H, W).astype(np.float32)


def normalizar_contraste(volume: np.ndarray, p_bajo: float, p_alto: float):
    """
    Estira el rango dinámico del volumen al rango [0, 1].
    Usa solo los píxeles de tejido (por encima del 10% del máximo) para
    calcular los percentiles, evitando que el fondo interpolado distorsione
    el cálculo.
    Retorna (volume_normalizado, v_min, v_max) para guardar en metadatos.
    """
    umbral_fondo = float(volume.max()) * 0.05   # ignora píxeles casi negros
    tejido = volume[volume > umbral_fondo]
    if tejido.size == 0:
        return volume, 0.0, float(volume.max())
    v_min = float(np.percentile(tejido, p_bajo))
    v_max = float(np.percentile(tejido, p_alto))
    if v_max <= v_min:
        return volume, v_min, v_max
    out = (volume - v_min) / (v_max - v_min)
    return np.clip(out, 0, 1), v_min, v_max


# ============================================================
# 1. CÁLCULO DE DIMENSIONES MÁXIMAS (una sola lectura de cabecera)
# ============================================================
print("Calculando dimensiones del lienzo final...")
max_h, max_w = 0, 0
for canal in canales:
    ruta_canal = os.path.join(ruta_raiz, canal)
    if not os.path.exists(ruta_canal):
        continue
    for f in os.listdir(ruta_canal):
        if not f.lower().endswith(".tif"):
            continue
        img_temp = io.imread(os.path.join(ruta_canal, f))
        h, w = img_temp.shape[:2]
        max_h = max(max_h, int(h * escala_reduccion))
        max_w = max(max_w, int(w * escala_reduccion))

max_h += 2
max_w += 2
print(f"Lienzo final: {max_h} x {max_w} píxeles")

# ============================================================
# 2. PROCESADO E INTERPOLACIÓN POR CANAL
# ============================================================
z_min, z_max   = min(indices_reales), max(indices_reales)
total_z_micras = (z_max - z_min) * micras_entre_indices + 1

volumenes      = {}   # canal -> array uint16 (Z, H, W)
contrast_store = {}   # canal -> [v_min_real, v_max_real] antes de normalizar

for canal in canales:
    ruta_canal = os.path.join(ruta_raiz, canal)
    if not os.path.exists(ruta_canal):
        print(f"[AVISO] No se encontró la carpeta: {ruta_canal}")
        continue

    archivos_carpeta = os.listdir(ruta_canal)
    print(f"\n{'='*50}\nProcesando canal: {canal}")

    # --- Carga y redimensionado ---
    cortes_dict = {}
    for n in indices_reales:
        match = [f for f in archivos_carpeta
                 if f.lower().startswith(f"c{n}_") or f.lower().startswith(f"c{n}-")]
        if not match:
            # fallback más permisivo
            match = [f for f in archivos_carpeta if f.lower().startswith(f"c{n}")]
        if match:
            img = io.imread(os.path.join(ruta_canal, match[0]))
            # Redimensionar conservando rango (float64 → rescale → float32)
            img_res = transform.rescale(
                img.astype(np.float32),
                escala_reduccion,
                preserve_range=True,
                anti_aliasing=True,
                channel_axis=None,
            )
            pos_z = (n - z_min) * micras_entre_indices
            cortes_dict[pos_z] = normalizar_tamano(img_res, max_h, max_w)

    if len(cortes_dict) < 2:
        print(f"  [ERROR] Se necesitan al menos 2 cortes para interpolar. Se omite el canal.")
        continue

    # --- Interpolación spline cúbico (vectorizada) ---
    print(f"  Interpolando {len(cortes_dict)} cortes con spline cúbico...")
    vol_f32 = interpolar_spline_cubico(cortes_dict, total_z_micras)
    print(f"  Shape del volumen: {vol_f32.shape}  |  dtype: {vol_f32.dtype}")

    # --- Suavizado gaussiano en Z para reducir banding ---
    if sigma_z > 0:
        print(f"  Aplicando suavizado gaussiano en Z (sigma={sigma_z})...")
        gaussian_filter(vol_f32, sigma=(sigma_z, 0, 0), output=vol_f32)

    # --- Normalización de contraste ---
    if NORMALIZAR_CONTRASTE:
        print(f"  Normalizando contraste (p{PERCENTIL_BAJO}–p{PERCENTIL_ALTO})...")
        vol_f32, v_min_real, v_max_real = normalizar_contraste(vol_f32, PERCENTIL_BAJO, PERCENTIL_ALTO)
        print(f"    Rango original de tejido: [{v_min_real:.1f}, {v_max_real:.1f}]")
        # Escalar a rango uint16 completo para máximo detalle
        vol_u16 = (vol_f32 * 65535).astype(np.uint16)
        contrast_store[canal] = [0, 65535]
    else:
        vol_u16 = np.clip(vol_f32, 0, 65535).astype(np.uint16)
        contrast_store[canal] = [0, 300]

    volumenes[canal] = vol_u16
    print(f"  ✓ Canal '{canal}' listo.")

# ============================================================
# 3. GUARDAR RESULTADOS
# ============================================================
# contrast_limits por canal: shape (n_canales, 2)
cl_array = np.array([
    contrast_store.get(c, [0, 65535]) for c in canales
], dtype=np.float32)

metadatos = {
    "pixel_size":       np.float64(pixel_size),
    "escala_reduccion": np.float64(escala_reduccion),
    "canales":          np.array(canales),
    "colores":          np.array(["blue", "green", "red"]),
    "contrast_limits":  cl_array,          # shape (3, 2)
}

print(f"\nGuardando volúmenes en: {archivo_salida}")
np.savez_compressed(
    archivo_salida,
    **{f"vol_{c.replace('-', '_')}": v for c, v in volumenes.items()},
    **metadatos,
)

print("✓ Guardado completado.")
print(f"  Archivo: {archivo_salida}")
print(f"  Canales guardados: {list(volumenes.keys())}")
