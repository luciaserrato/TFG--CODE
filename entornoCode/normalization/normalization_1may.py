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
import time
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
GFAP_UMBRAL_PERCENTIL = 70   # percentil sobre señal real (mitad superior)
GFAP_UMBRAL_ABSOLUTO  = 40   # umbral mínimo absoluto — ajusta entre 30-80 según napari
GFAP_EROSION_PX       = 1    # erosión mínima
GFAP_APERTURA_PX      = 2    # apertura suave
GFAP_CIERRE_PX        = 25   # cierre agresivo para rellenar huecos entre cortes
GFAP_SOLO_COMPONENTE  = True # conservar solo el componente conectado mayor

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
    Interpolación cúbica Z para densificar el volumen.

    Usa scipy.interpolate.CubicSpline por columna para generar transiciones
    suaves entre cortes reales, eliminando las rayas/bandas horizontales
    que aparecen con interpolación lineal cuando los cortes están muy separados.

    vol_sparse : array (N_cortes, H, W) con los cortes originales
    Retorna    : array (N_planos_densos, H, W) uint16
    """
    from scipy.interpolate import CubicSpline

    pasos   = int(round(micras_entre_cortes / micras_entre_indices))
    N, H, W = vol_sparse.shape
    N_denso = (N - 1) * pasos + 1

    # Posiciones Z de los cortes reales (en µm)
    z_reales = np.arange(N) * micras_entre_cortes
    # Posiciones Z del volumen denso
    z_denso  = np.linspace(0, z_reales[-1], N_denso)

    print(f"    Interpolación cúbica: {N} cortes → {N_denso} planos")

    vol_denso = np.zeros((N_denso, H, W), dtype=np.float32)

    # Interpolar columna a columna en bloques para eficiencia de memoria
    BLOQUE = 20  # filas por bloque
    for y0 in range(0, H, BLOQUE):
        y1 = min(y0 + BLOQUE, H)
        bloque = vol_sparse[:, y0:y1, :]   # (N, bloque, W)

        # CubicSpline espera (N, n_puntos) — aplanamos (bloque, W)
        datos_2d = bloque.reshape(N, -1).astype(np.float64)
        cs = CubicSpline(z_reales, datos_2d, axis=0, bc_type="not-a-knot")
        interpolado = cs(z_denso)   # (N_denso, bloque*W)

        vol_denso[:, y0:y1, :] = interpolado.reshape(N_denso, y1 - y0, W)

        if y0 % 100 == 0:
            print(f"      fila {y0}/{H}...")

    # Clamp: la cúbica puede generar valores negativos
    vol_denso = vol_denso.clip(0)

    # Suavizado gaussiano Z: sigma=20 planos = 20 µm
    # Cubre exactamente la mitad del hueco de 40 µm entre cortes reales
    print("    Suavizando en Z (sigma=20 planos)...")
    vol_denso = ndimage.gaussian_filter1d(vol_denso, sigma=20, axis=0)

    return vol_denso.clip(0).astype(np.uint16)


def segmentar_gfap(vol_gfap):
    """
    Segmentación morfológica 3D para aislar la masa tumoral GFAP.

    PROBLEMA CLAVE: vol_gfap es un volumen interpolado con 1 plano/µm.
    Entre cortes reales (separados 40 µm = 40 planos) hay valores bajos
    de interpolación. Si calculamos el percentil sobre TODO el volumen,
    los planos interpolados casi-cero distorsionan el umbral.

    SOLUCIÓN: calcular el umbral solo sobre los píxeles del percentil
    superior de cada plano Z con señal real, no sobre todo el volumen.
    """
    print(f"  Segmentando GFAP → umbral p{GFAP_UMBRAL_PERCENTIL}, "
          f"erosion={GFAP_EROSION_PX}, apertura={GFAP_APERTURA_PX}, "
          f"cierre={GFAP_CIERRE_PX}")

    # El GFAP tiene señal difusa en ~99% del tejido (fondo astrocítico).
    # El tumor es solo la región MÁS BRILLANTE.
    # Estrategia: umbral = max(GFAP_UMBRAL_ABSOLUTO, percentil alto sobre señal real)
    # Así siempre cortamos el fondo difuso aunque el percentil lo baje.
    todos_positivos = vol_gfap[vol_gfap > 0]
    if todos_positivos.size == 0:
        return np.zeros_like(vol_gfap)

    mediana    = float(np.percentile(todos_positivos, 50))
    señal_real = todos_positivos[todos_positivos > mediana]
    umbral_pct = float(np.percentile(señal_real, GFAP_UMBRAL_PERCENTIL))

    # Tomar el mayor entre el percentil calculado y el umbral absoluto mínimo
    umbral = max(umbral_pct, float(GFAP_UMBRAL_ABSOLUTO))

    mascara = (vol_gfap >= umbral).astype(np.uint8)
    print(f"    Mediana: {mediana:.1f}  Umbral percentil: {umbral_pct:.1f}  "
          f"Umbral absoluto: {GFAP_UMBRAL_ABSOLUTO}  → Umbral final: {umbral:.1f}")
    print(f"    Vóxeles iniciales: {mascara.sum():,}")

    struct = ndimage.generate_binary_structure(3, 1)

    if GFAP_EROSION_PX > 0:
        mascara = ndimage.binary_erosion(
            mascara, structure=struct,
            iterations=GFAP_EROSION_PX).astype(np.uint8)
        print(f"    Tras erosión:   {mascara.sum():,} vóxeles")

    if GFAP_APERTURA_PX > 0:
        mascara = ndimage.binary_opening(
            mascara, structure=struct,
            iterations=GFAP_APERTURA_PX).astype(np.uint8)
        print(f"    Tras apertura:  {mascara.sum():,} vóxeles")

    if GFAP_CIERRE_PX > 0:
        mascara = ndimage.binary_closing(
            mascara, structure=struct,
            iterations=GFAP_CIERRE_PX).astype(np.uint8)
        print(f"    Tras cierre:    {mascara.sum():,} vóxeles")

    if GFAP_SOLO_COMPONENTE:
        etiquetas, n_comp = ndimage.label(mascara)
        if n_comp > 1:
            tamaños = ndimage.sum(mascara, etiquetas, range(1, n_comp + 1))
            mayor   = int(np.argmax(tamaños)) + 1
            mascara = (etiquetas == mayor).astype(np.uint8)
            print(f"    Componentes: {n_comp} → conservado el mayor "
                  f"({tamaños[mayor-1]:,.0f} vóxeles)")

    vol_enmascarado = (vol_gfap * mascara).astype(np.uint16)
    print(f"    ✓ Segmentación final: {mascara.sum():,} vóxeles")
    return vol_enmascarado


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def main():
    carpeta = Path(CARPETA_ENTRADA)
    salida  = Path(ARCHIVO_SALIDA)
    salida.parent.mkdir(parents=True, exist_ok=True)

    xy_um_por_pixel = PIXEL_SIZE_UM / ESCALA_REDUCCION
    z_um_por_plano  = MICRAS_ENTRE_INDICES

    t_inicio = time.time()
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
        t_interp = time.time()
        vol_denso = interpolar_z(vol_sparse_ali, MICRAS_ENTRE_CORTES, MICRAS_ENTRE_INDICES)
        print(f"  Interpolación completada en {time.time()-t_interp:.1f} s")
        print(f"  Volumen denso: {vol_denso.shape}  dtype={vol_denso.dtype}")

        volumenes[canal] = vol_denso

    # ── Segmentación GFAP desactivada ────────────────────────
    # El marcaje GFAP es débil y difuso — la segmentación morfológica
    # destruye la señal. Se guarda el volumen completo y la segmentación
    # se hace en tiempo real desde el visualizador con el widget.
    print("\n  [INFO] Segmentación GFAP omitida — se usará el widget en napari")

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
    t_total = time.time() - t_inicio
    print(f"  Tiempo total: {t_total/60:.1f} min ({t_total:.0f} s)")
    print("=" * 60)


if __name__ == "__main__":
    main()