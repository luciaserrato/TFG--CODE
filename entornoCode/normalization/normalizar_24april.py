import os
import re
import cv2
import numpy as np
from skimage import io, transform
from scipy import ndimage

# ============================================================
# CONFIGURACIÓN
# ============================================================
pixel_size           = 1.1364
micras_entre_indices = 40
indices_reales       = [45, 47, 49, 51, 52, 54, 56, 58, 60, 62, 64, 67, 68, 70, 71, 73]

ruta_raiz        = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\Reconstruccion_v2"
canales          = ["C00-DAPI", "C01-IBA-1", "C02-GFAP"]
escala_reduccion = 0.2

archivo_salida = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\volumenes\volumenes.npz"

# ============================================================
# PARÁMETROS DE LA MÁSCARA GFAP
# Ajusta estos valores si la máscara recorta demasiado o demasiado poco.
#
# GFAP_UMBRAL_PERCENTIL: percentil sobre píxeles no negros usado como
#   umbral de corte. Sube el valor (ej: 85) para quedarte solo con las
#   zonas más brillantes (núcleo tumoral). Bájalo (ej: 60) para incluir
#   más tejido reactivo.
#
# GFAP_APERTURA_PX: radio en píxeles del filtro de apertura morfológica
#   que elimina pequeños objetos aislados (ruido puntual). Auméntalo si
#   quedan muchos puntos sueltos fuera del tumor.
#
# GFAP_CIERRE_PX: radio en píxeles del filtro de cierre morfológico
#   que rellena huecos dentro de la masa tumoral. Auméntalo si el tumor
#   aparece fragmentado internamente.
# ============================================================
GFAP_UMBRAL_PERCENTIL = 75   # [0-100] — sube para aislar mejor el núcleo
GFAP_APERTURA_PX      = 3    # elimina objetos pequeños fuera del tumor
GFAP_CIERRE_PX        = 8    # rellena huecos dentro del tumor


# ============================================================
# FUNCIONES DE APOYO
# ============================================================
def normalizar_tamano(img, target_h, target_w):
    """Ajusta la imagen al lienzo target. Si es más pequeña, rellena con negro."""
    h, w = img.shape[:2]
    nuevo_lienzo = np.zeros((target_h, target_w), dtype=np.uint16)
    h_lim = min(h, target_h)
    w_lim = min(w, target_w)
    nuevo_lienzo[:h_lim, :w_lim] = img[:h_lim, :w_lim]
    return nuevo_lienzo


def buscar_archivo_corte(archivos_carpeta, n):
    """
    Búsqueda robusta del archivo para el índice n.
    Usa regex para evitar que c4 haga match con c47.
    """
    patron = re.compile(rf"^c0*{n}[_.\-]|^0*{n}[_.\-]", re.IGNORECASE)
    match = [f for f in archivos_carpeta if patron.match(f)]
    if not match:
        match = [f for f in archivos_carpeta
                 if os.path.splitext(f)[0] == str(n) or
                    os.path.splitext(f)[0].lower() == f"c{n}"]
    return match


def registrar_ecc(img_ref, img_mov):
    """
    Alinea img_mov sobre img_ref con Enhanced Correlation Coefficient.
    Robusto ante diferencias de brillo entre cortes consecutivos de criostato.
    Si ECC no converge, devuelve img_mov sin modificar.
    """
    ref_8 = cv2.normalize(img_ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mov_8 = cv2.normalize(img_mov, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criterios   = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)

    try:
        _, warp_matrix = cv2.findTransformECC(
            ref_8, mov_8, warp_matrix,
            cv2.MOTION_EUCLIDEAN, criterios,
            inputMask=None, gaussFiltSize=5
        )
        h, w         = img_mov.shape
        img_alineada = cv2.warpAffine(
            img_mov, warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        return img_alineada.astype(np.uint16)
    except cv2.error:
        print("    [AVISO] ECC no convergió → corte sin alinear")
        return img_mov


def calcular_contraste(vol, p_low=1, p_high=99):
    """Percentiles sobre píxeles no negros. Más robusto que [0, 300] fijo."""
    validos = vol[vol > 0]
    if validos.size == 0:
        return [0.0, 1.0]
    lo = float(np.percentile(validos, p_low))
    hi = float(np.percentile(validos, p_high))
    return [lo, max(hi, lo + 1)]


def aplicar_mascara_gfap(vol_gfap, umbral_percentil, apertura_px, cierre_px):
    """
    Genera una versión enmascarada del volumen GFAP que conserva solo
    las zonas de alta intensidad (núcleo tumoral) y suprime el fondo reactivo.

    Estrategia:
      1. Umbral por percentil → máscara binaria inicial
      2. Apertura morfológica  → elimina puntos aislados (ruido)
      3. Cierre morfológico    → rellena huecos dentro del tumor
      4. Aplicar máscara al volumen original → conserva intensidades reales

    El volumen original sin máscara también se guarda en el npz para que
    el visualizador pueda alternar entre ambos sin necesidad de reejecutar
    este script.
    """
    print(f"  Generando máscara GFAP (percentil={umbral_percentil}, "
          f"apertura={apertura_px}px, cierre={cierre_px}px)...")

    validos = vol_gfap[vol_gfap > 0]
    if validos.size == 0:
        return vol_gfap.copy()

    umbral  = float(np.percentile(validos, umbral_percentil))
    mascara = (vol_gfap >= umbral).astype(np.uint8)

    # Elemento estructurante esférico aproximado en 3D
    radio = max(1, apertura_px)
    struct = ndimage.generate_binary_structure(3, 1)

    # Apertura: elimina objetos pequeños (ruido puntual fuera del tumor)
    mascara = ndimage.binary_opening(mascara, structure=struct,
                                     iterations=radio).astype(np.uint8)

    # Cierre: rellena huecos dentro de la masa tumoral
    radio_c = max(1, cierre_px)
    mascara = ndimage.binary_closing(mascara, structure=struct,
                                     iterations=radio_c).astype(np.uint8)

    vol_enmascarado = (vol_gfap * mascara).astype(np.uint16)

    n_voxels_tumor  = int(mascara.sum())
    n_voxels_total  = int((vol_gfap > 0).sum())
    print(f"  Vóxeles conservados: {n_voxels_tumor:,} / {n_voxels_total:,} "
          f"({100*n_voxels_tumor/max(n_voxels_total,1):.1f}%)")

    return vol_enmascarado


# ============================================================
# 1. CÁLCULO DE DIMENSIONES MÁXIMAS
# ============================================================
print("Calculando dimensiones del lienzo final...")
max_h, max_w = 0, 0
for canal in canales:
    ruta_canal = os.path.join(ruta_raiz, canal)
    if not os.path.exists(ruta_canal):
        continue
    archivos = [f for f in os.listdir(ruta_canal) if f.lower().endswith(".tif")]
    for f in archivos:
        img_temp = io.imread(os.path.join(ruta_canal, f))
        h, w     = img_temp.shape[:2]
        max_h    = max(max_h, int(h * escala_reduccion))
        max_w    = max(max_w, int(w * escala_reduccion))

max_h += 2
max_w += 2
print(f"Lienzo final: {max_h} x {max_w}")

# ============================================================
# 2. PROCESADO E INTERPOLACIÓN POR CANAL
# ============================================================
z_min, z_max   = min(indices_reales), max(indices_reales)
total_z_micras = (z_max - z_min) * micras_entre_indices + 1

volumenes = {}  # nombre_canal -> array 3D

for canal in canales:
    ruta_canal = os.path.join(ruta_raiz, canal)
    if not os.path.exists(ruta_canal):
        print(f"[AVISO] No se encontró la carpeta: {ruta_canal}")
        continue

    cortes_dict      = {}
    archivos_carpeta = os.listdir(ruta_canal)
    print(f"\nProcesando canal: {canal}")

    for n in indices_reales:
        match = buscar_archivo_corte(archivos_carpeta, n)
        if match:
            img     = io.imread(os.path.join(ruta_canal, match[0]))
            img_res = transform.rescale(
                img, escala_reduccion,
                preserve_range=True, anti_aliasing=True
            ).astype(np.uint16)
            pos_z              = (n - z_min) * micras_entre_indices
            cortes_dict[pos_z] = normalizar_tamano(img_res, max_h, max_w)
        else:
            print(f"  [AVISO] No se encontró archivo para índice {n}")

    posiciones_reales = sorted(cortes_dict.keys())

    # Registro ECC: alinea cada corte respecto al anterior
    print(f"  Registrando {len(posiciones_reales)} cortes con ECC...")
    for i in range(1, len(posiciones_reales)):
        z_ref = posiciones_reales[i - 1]
        z_mov = posiciones_reales[i]
        cortes_dict[z_mov] = registrar_ecc(cortes_dict[z_ref], cortes_dict[z_mov])

    volume_3d = np.zeros((total_z_micras, max_h, max_w), dtype=np.uint16)

    print(f"  Interpolando {len(posiciones_reales)} cortes...")
    for i in range(len(posiciones_reales) - 1):
        z_ini, z_fin     = posiciones_reales[i], posiciones_reales[i + 1]
        img_ini, img_fin = cortes_dict[z_ini], cortes_dict[z_fin]
        distancia        = z_fin - z_ini

        for p in range(distancia + 1):
            z_actual = z_ini + p
            if z_actual >= total_z_micras:
                break
            peso_fin           = p / distancia
            peso_ini           = 1 - peso_fin
            volume_3d[z_actual] = (img_ini * peso_ini + img_fin * peso_fin).astype(np.uint16)

    # Último corte real
    z_ultimo = posiciones_reales[-1]
    if z_ultimo < total_z_micras:
        volume_3d[z_ultimo] = cortes_dict[z_ultimo]

    volumenes[canal] = volume_3d
    print(f"  Canal '{canal}' procesado. Shape: {volume_3d.shape}")

# ============================================================
# 3. MÁSCARA GFAP
# ============================================================
canal_gfap = "C02-GFAP"
if canal_gfap in volumenes:
    volumenes[canal_gfap + "_mascara"] = aplicar_mascara_gfap(
        volumenes[canal_gfap],
        GFAP_UMBRAL_PERCENTIL,
        GFAP_APERTURA_PX,
        GFAP_CIERRE_PX,
    )

# ============================================================
# 4. GUARDAR RESULTADOS
# ============================================================

# Escala física correcta:
#   Z:  1 plano por µm interpolado → z_um_por_plano = 1.0
#       (pasar 40 aquí estiraría el volumen x40 en Z)
#   XY: píxel reducido = pixel_size / escala_reduccion µm
z_um_por_plano  = 1.0
xy_um_por_pixel = pixel_size / escala_reduccion

# Contraste automático (percentil 1-99 sobre píxeles no negros)
# Se calcula sobre todos los canales incluyendo la máscara GFAP
canales_a_guardar = [c for c in list(volumenes.keys())]
contrast_limits   = np.array([
    calcular_contraste(volumenes[c]) for c in canales_a_guardar
])  # shape (n_canales, 2)

metadatos = {
    "pixel_size":            np.float64(pixel_size),
    "escala_reduccion":      np.float64(escala_reduccion),
    "z_um_por_plano":        np.float64(z_um_por_plano),
    "xy_um_por_pixel":       np.float64(xy_um_por_pixel),
    "canales":               np.array(canales),
    "canales_guardados":     np.array(canales_a_guardar),  # incluye _mascara
    "colores":               np.array(["blue", "green", "red"]),
    "contrast_limits":       contrast_limits,
    "gfap_umbral_percentil": np.float64(GFAP_UMBRAL_PERCENTIL),
    "gfap_apertura_px":      np.float64(GFAP_APERTURA_PX),
    "gfap_cierre_px":        np.float64(GFAP_CIERRE_PX),
}

print(f"\nMetadatos de escala:")
print(f"  z_um_por_plano  = {z_um_por_plano} µm/plano")
print(f"  xy_um_por_pixel = {xy_um_por_pixel:.4f} µm/píxel")
print(f"  contrast_limits = {contrast_limits.tolist()}")

os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
print(f"\nGuardando volúmenes en: {archivo_salida}")
np.savez_compressed(
    archivo_salida,
    **{f"vol_{c.replace('-', '_')}": v for c, v in volumenes.items()},
    **metadatos,
)

print("✓ Guardado completado.")
print(f"  Archivo:           {archivo_salida}")
print(f"  Canales guardados: {canales_a_guardar}")