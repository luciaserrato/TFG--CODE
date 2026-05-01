"""
normalizar_tfg_v2.py
====================
Pipeline de normalización 3D para reconstrucción de neuroblastoma.

Mejoras respecto a versión anterior:
- Alineamiento ECC más robusto con manejo de fallos de convergencia
- Interpolación Z suavizada para morfología tumoral más compacta
- Segmentación GFAP más agresiva (erosión + componente mayor)
- Gestión eficiente de memoria con bloques y dtypes reducidos
- Metadatos completos guardados en el .npz para el visualizador
"""

import numpy as np
import cv2
from scipy import ndimage
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN — AJUSTA ESTOS PARÁMETROS
# ============================================================
# Carpeta raíz que contiene las subcarpetas C00-DAPI, C01-IBA-1, C02-GFAP
CARPETA_ENTRADA   = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\Reconstruccion_v2"
ARCHIVO_SALIDA    = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\volumenes\volumenes.npz"

# Escala y física
PIXEL_SIZE_UM     = 1.1364   # µm por píxel original
ESCALA_REDUCCION  = 0.2      # factor de downscale XY (0.2 → 5x más pequeño)
MICRAS_ENTRE_CORTES = 40.0   # µm entre cortes reales de criostato
MICRAS_ENTRE_INDICES = 1.0   # µm entre planos interpolados (1 → 1 plano/µm)

# Canales (orden en que están guardados los TIFF)
CANALES = ["C00-DAPI", "C01-IBA-1", "C02-GFAP"]
COLORES = ["blue",     "green",     "red"]

# ── Alineamiento ECC ──────────────────────────────────────────
ECC_ITERATIONS    = 200      # máx iteraciones por par de cortes
ECC_EPSILON       = 1e-6     # tolerancia de convergencia
ECC_WARP_MODE     = cv2.MOTION_TRANSLATION  # solo traslación (más estable)
ECC_GAUSS_SIGMA   = 5        # suavizado previo al ECC para robustez

# ── Segmentación GFAP ─────────────────────────────────────────
GFAP_UMBRAL_PERCENTIL = 60   # reducido para incluir más señal
GFAP_UMBRAL_ABSOLUTO  = 30   # reducido para incluir más señal
GFAP_EROSION_PX       = 0    # eliminado para no perder señal
GFAP_APERTURA_PX      = 1    # reducido
GFAP_CIERRE_PX        = 2    # reducido para no colapsar Z
GFAP_SOLO_COMPONENTE  = False # desactivado para mantener todos los cortes

# Índices de corte válidos (número en el nombre del archivo, ej: c45 → 45)
# Deja la lista vacía [] para coger todos los TIFFs de la carpeta
INDICES_VALIDOS = [45, 47, 49, 51, 52, 54, 56, 58, 60, 62, 64, 67, 68, 70, 71, 73]

# ── Contraste ─────────────────────────────────────────────────
P_LOW  = 1
P_HIGH = 99

# ============================================================
# UTILIDADES
# ============================================================
def calcular_contraste(vol, p_low=P_LOW, p_high=P_HIGH):
    datos = vol[vol > 0]
    if datos.size == 0:
        return np.array([0.0, 1.0])
    lo = float(np.percentile(datos, p_low))
    hi = float(np.percentile(datos, p_high))
    return np.array([lo, max(hi, lo + 1)])


def reducir_imagen(img_2d, escala):
    """Downscale con interpolación por área (mejor calidad para reducción)."""
    h, w = img_2d.shape
    nh = max(1, int(round(h * escala)))
    nw = max(1, int(round(w * escala)))
    # cv2 espera uint8 o float32
    img_f = img_2d.astype(np.float32)
    return cv2.resize(img_f, (nw, nh), interpolation=cv2.INTER_AREA)


def alinear_cortes(ref_2d, mov_2d):
    """
    Alinea mov_2d sobre ref_2d en dos pasos:
      1. Phase correlation (robusto, rápido) → estima traslación gruesa
      2. ECC (opcional, refinamiento fino) → solo si la correlación es alta

    Para imágenes de fluorescencia con poco contraste global, la correlación
    de fase es mucho más estable que ECC puro.
    """
    def to_f32_norm(im):
        """Normaliza a float32 en [0, 1], aplica CLAHE para realzar contraste."""
        mn, mx = float(im.min()), float(im.max())
        if mx == mn:
            return np.zeros_like(im, dtype=np.float32)
        im_f = ((im.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
        # CLAHE: realza contraste local → mejora correlación en zonas poco brillantes
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        im_f  = clahe.apply(im_f)
        return im_f.astype(np.float32) / 255.0

    ref_n = to_f32_norm(ref_2d)
    mov_n = to_f32_norm(mov_2d)

    h, w = mov_2d.shape

    # ── Paso 1: Phase correlation ──────────────────────────────
    # Estima el desplazamiento (dx, dy) entre los dos cortes
    try:
        (dx, dy), respuesta = cv2.phaseCorrelate(ref_n, mov_n)
        # respuesta: valor en [0,1] que indica calidad de la correlación
        # Si es muy baja, las imágenes no se solapan bien → no alinear
        if respuesta < 0.02:
            return mov_2d.astype(np.float32)  # sin alinear

        # Limitar desplazamiento máximo razonable (10% del tamaño)
        max_dx = w * 0.10
        max_dy = h * 0.10
        if abs(dx) > max_dx or abs(dy) > max_dy:
            print(f"    [ALIGN] Desplazamiento excesivo ({dx:.1f}, {dy:.1f}) → ignorado")
            return mov_2d.astype(np.float32)

        # Aplicar traslación
        M_traslacion = np.float32([[1, 0, -dx], [0, 1, -dy]])
        alineada = cv2.warpAffine(
            mov_2d.astype(np.float32), M_traslacion, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE)

    except cv2.error:
        return mov_2d.astype(np.float32)

    # ── Paso 2: Refinamiento ECC (solo si la correlación fue buena) ──
    if respuesta > 0.10 and ECC_ITERATIONS > 0:
        ref_u8 = (ref_n * 255).astype(np.uint8)
        ali_u8 = cv2.normalize(alineada, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if ECC_GAUSS_SIGMA > 0:
            ref_u8 = cv2.GaussianBlur(ref_u8, (0, 0), ECC_GAUSS_SIGMA)
            ali_u8 = cv2.GaussianBlur(ali_u8, (0, 0), ECC_GAUSS_SIGMA)
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    ECC_ITERATIONS, ECC_EPSILON)
        try:
            _, warp = cv2.findTransformECC(ref_u8, ali_u8, warp,
                                           cv2.MOTION_TRANSLATION, criteria)
            alineada = cv2.warpAffine(
                alineada, warp, (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE)
        except cv2.error:
            pass  # se queda con la traslación gruesa de phase correlation

    return alineada


def interpolar_z(vol_sparse, micras_entre_cortes, micras_entre_indices):
    """
    Interpolación lineal Z para densificar el volumen.

    vol_sparse : array (N_cortes, H, W) con los cortes originales
    Retorna    : array (N_planos_densos, H, W) uint16
    """
    pasos = int(round(micras_entre_cortes / micras_entre_indices))
    N, H, W = vol_sparse.shape
    N_denso = (N - 1) * pasos + 1

    vol_denso = np.zeros((N_denso, H, W), dtype=np.float32)
    for i in range(N - 1):
        for p in range(pasos):
            alpha = p / pasos
            idx   = i * pasos + p
            vol_denso[idx] = (1 - alpha) * vol_sparse[i] + alpha * vol_sparse[i + 1]
    vol_denso[(N - 1) * pasos] = vol_sparse[-1]

    # Suavizado Z leve post-interpolación (sigma=0.8) para reducir artefactos
    vol_denso = ndimage.gaussian_filter1d(vol_denso, sigma=0.8, axis=0)

    return vol_denso.clip(0).astype(np.uint16)


def segmentar_gfap(vol_gfap):
    """
    Segmentación GFAP SIMPLE y estable (no colapsa Z).

    - Umbral global (sin lógica rara por plano)
    - Morfología suave opcional
    - Mantiene estructura 3D
    """

    print(f"  Segmentando GFAP (simple) → p{GFAP_UMBRAL_PERCENTIL}")

    if vol_gfap.ndim != 3:
        raise ValueError("GFAP debe ser volumen 3D (Z,H,W)")

    # =========================
    # UMBRAL GLOBAL SIMPLE
    # =========================
    positivos = vol_gfap[vol_gfap > 0]

    if positivos.size == 0:
        return np.zeros_like(vol_gfap, dtype=np.uint8)

    umbral = np.percentile(positivos, GFAP_UMBRAL_PERCENTIL)
    umbral = max(umbral, GFAP_UMBRAL_ABSOLUTO)

    mascara = (vol_gfap >= umbral).astype(np.uint8)

    print(f"    Umbral: {umbral:.2f}")
    print(f"    Voxels iniciales: {mascara.sum():,}")

    # =========================
    # MORFOLOGÍA SUAVE (NO ROMPE Z)
    # =========================
    struct = ndimage.generate_binary_structure(3, 1)

    if GFAP_APERTURA_PX > 0:
        mascara = ndimage.binary_opening(
            mascara, structure=struct,
            iterations=GFAP_APERTURA_PX).astype(np.uint8)

    # ⚠️ cierre MUY moderado
    if GFAP_CIERRE_PX > 0:
        mascara = ndimage.binary_closing(
            mascara, structure=struct,
            iterations=min(2, GFAP_CIERRE_PX)).astype(np.uint8)

    print(f"    Tras morfología: {mascara.sum():,}")

    # =========================
    # COMPONENTE MAYOR (OPCIONAL)
    # =========================
    if GFAP_SOLO_COMPONENTE:
        etiquetas, n_comp = ndimage.label(mascara)

        if n_comp > 1:
            tamaños = ndimage.sum(mascara, etiquetas, range(1, n_comp + 1))
            mayor = int(np.argmax(tamaños)) + 1
            mascara = (etiquetas == mayor).astype(np.uint8)

            print(f"    Componentes: {n_comp} → mayor: {tamaños[mayor-1]:,.0f}")

    print("    Shape:", mascara.shape)
    print("    Slices activas:",
          sum(mascara[z].sum() > 0 for z in range(mascara.shape[0])))

    print(f"    ✓ Segmentación final: {mascara.sum():,}")

    return mascara.astype(np.uint8)


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def main():
    carpeta = Path(CARPETA_ENTRADA)
    salida  = Path(ARCHIVO_SALIDA)
    salida.parent.mkdir(parents=True, exist_ok=True)

    xy_um_por_pixel = PIXEL_SIZE_UM / ESCALA_REDUCCION
    z_um_por_plano  = MICRAS_ENTRE_INDICES

    print("=" * 60)
    print("PIPELINE DE NORMALIZACIÓN — TFG NEUROBLASTOMA")
    print("=" * 60)
    print(f"Entrada:  {carpeta}")
    print(f"Salida:   {salida}")
    print(f"Escala XY: {xy_um_por_pixel:.4f} µm/px  |  Z: {z_um_por_plano} µm/plano")

    # ── Cargar y reducir cortes por canal ─────────────────────
    volumenes = {}

    for canal, color in zip(CANALES, COLORES):
        print(f"\n{'─'*50}")
        print(f"Canal: {canal}")

        # Cada canal tiene su propia subcarpeta (C00-DAPI, C01-IBA-1, C02-GFAP)
        subcarpeta = carpeta / canal
        if not subcarpeta.exists():
            print(f"  [AVISO] Subcarpeta no encontrada: {subcarpeta}")
            continue

        # Recoger todos los TIFFs y ordenar numéricamente por el número del corte
        # (los nombres son del tipo c45_IBA-1_Final.tif → extrae el número)
        import re
        def extraer_numero(p):
            m = re.search(r'(\d+)', p.stem)
            return int(m.group(1)) if m else 0

        patron = sorted(
            list(subcarpeta.glob("*.tif")) + list(subcarpeta.glob("*.tiff")),
            key=extraer_numero
        )
        if not patron:
            print(f"  [AVISO] No se encontraron TIFFs en: {subcarpeta}")
            continue

        # Filtrar solo los índices válidos si se ha definido la lista
        if INDICES_VALIDOS:
            patron = [p for p in patron if extraer_numero(p) in INDICES_VALIDOS]
            if not patron:
                print(f"  [AVISO] Ningún TIFF coincide con INDICES_VALIDOS en: {subcarpeta}")
                continue

        print(f"  Subcarpeta: {subcarpeta}")
        print(f"  Encontrados {len(patron)} cortes")
        print(f"  Primer corte: {patron[0].name}  |  Último: {patron[-1].name}")

        # Leer y reducir (en float32 para evitar overflow en operaciones)
        # El tamaño de referencia lo fija el primer corte válido;
        # los demás se ajustan al mismo tamaño para poder apilarlos.
        cortes    = []
        shape_ref = None   # (H, W) tras reducción

        for ruta in patron:
            img = cv2.imread(str(ruta), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"  [AVISO] No se pudo leer: {ruta.name}")
                continue
            if img.ndim == 3:
                img = img[:, :, 0]
            img_red = reducir_imagen(img, ESCALA_REDUCCION)

            if shape_ref is None:
                shape_ref = img_red.shape   # (H, W) de referencia
                print(f"  Tamaño de referencia (reducido): {shape_ref}")

            # Si el corte tiene distinto tamaño, redimensionar al de referencia
            if img_red.shape != shape_ref:
                print(f"  [INFO] {ruta.name} -> {img_red.shape} != {shape_ref}, ajustando...")
                img_red = cv2.resize(img_red, (shape_ref[1], shape_ref[0]),
                                     interpolation=cv2.INTER_AREA)
            cortes.append(img_red)

        if not cortes:
            continue

        vol_sparse = np.stack(cortes, axis=0)   # (N, H, W) float32
        print(f"  Volumen sparse: {vol_sparse.shape}  dtype={vol_sparse.dtype}")

        # ── Alineamiento ECC ──────────────────────────────────
        print(f"  Alineando cortes con ECC...")
        vol_alineado = [vol_sparse[0]]
        for i in range(1, len(vol_sparse)):
            ref = vol_alineado[-1]
            mov = vol_sparse[i]
            ali = alinear_cortes(ref, mov)
            vol_alineado.append(ali)
            if (i + 1) % 10 == 0:
                print(f"    Procesado corte {i+1}/{len(vol_sparse)}")
        vol_sparse_ali = np.stack(vol_alineado, axis=0).astype(np.float32)

        # ── Interpolación Z ───────────────────────────────────
        print(f"  Interpolando Z ({MICRAS_ENTRE_CORTES} µm → {MICRAS_ENTRE_INDICES} µm/plano)...")
        vol_denso = interpolar_z(vol_sparse_ali, MICRAS_ENTRE_CORTES, MICRAS_ENTRE_INDICES)
        print(f"  Volumen denso: {vol_denso.shape}  dtype={vol_denso.dtype}")

        volumenes[canal] = vol_denso

    # ── Segmentación GFAP ─────────────────────────────────────
    canal_gfap = "C02-GFAP"
    if canal_gfap in volumenes:
        print(f"\n{'─'*50}")
        print("Segmentando masa tumoral (GFAP)...")
        vol_gfap_mask = segmentar_gfap(volumenes[canal_gfap])
        volumenes["C02-GFAP_mascara"] = vol_gfap_mask
    else:
        print(f"\n[AVISO] Canal {canal_gfap} no disponible → saltando segmentación")

    # ── Calcular contrastes ───────────────────────────────────
    canales_guardados = []
    contrast_limits   = []

    todos_canales = CANALES + (["C02-GFAP_mascara"] if "C02-GFAP_mascara" in volumenes else [])
    for canal in todos_canales:
        if canal in volumenes:
            cl = calcular_contraste(volumenes[canal])
            canales_guardados.append(canal)
            contrast_limits.append(cl)

    # ── Guardar .npz ──────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Guardando en: {salida}")

    datos_guardar = {
        # Metadatos de escala
        "pixel_size":           np.float64(PIXEL_SIZE_UM),
        "escala_reduccion":     np.float64(ESCALA_REDUCCION),
        "z_um_por_plano":       np.float64(z_um_por_plano),
        "xy_um_por_pixel":      np.float64(xy_um_por_pixel),

        # Metadatos de canales (guardados como strings UTF-8, sin object dtype)
        "canales":              np.array(CANALES),
        "colores":              np.array(COLORES),
        "canales_guardados":    np.array(canales_guardados),
        "contrast_limits":      np.array(contrast_limits, dtype=np.float64),

        # Parámetros de segmentación (para reproducir en visualizador)
        "gfap_umbral_percentil": np.int32(GFAP_UMBRAL_PERCENTIL),
        "gfap_erosion_px":       np.int32(GFAP_EROSION_PX),
        "gfap_apertura_px":      np.int32(GFAP_APERTURA_PX),
        "gfap_cierre_px":        np.int32(GFAP_CIERRE_PX),
    }

    # Agregar volúmenes
    for canal, vol in volumenes.items():
        clave = f"vol_{canal.replace('-', '_')}"
        datos_guardar[clave] = vol
        print(f"  {clave}: {vol.shape}  {vol.dtype}  "
              f"({vol.nbytes / 1e6:.1f} MB)")

    np.savez_compressed(str(salida), **datos_guardar)

    print("\n" + "=" * 60)
    print("✓ NORMALIZACIÓN COMPLETADA")
    print(f"  Archivo: {salida}")
    print(f"  Canales guardados: {canales_guardados}")
    print(f"  Escala Z:  {z_um_por_plano} µm/plano")
    print(f"  Escala XY: {xy_um_por_pixel:.4f} µm/píxel")
    print("=" * 60)


if __name__ == "__main__":
    main()