"""
visualizar_tfg_v2.py
====================
Visualización 3D del neuroblastoma con Napari.
"""

import numpy as np
import napari
from magicgui import magicgui
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN
# ============================================================
ARCHIVO_ENTRADA = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\volumenes\volumenes.npz"

P_LOW  = 1
P_HIGH = 99

# ============================================================
# 1. CARGAR DATOS Y METADATOS
# ============================================================
print(f"Cargando: {ARCHIVO_ENTRADA}")
datos = np.load(ARCHIVO_ENTRADA, allow_pickle=True)

canales = list(datos["canales"])
colores = list(datos["colores"])

z_um_por_plano  = float(datos["z_um_por_plano"])
xy_um_por_pixel = float(datos["xy_um_por_pixel"])
scale_xyz       = (z_um_por_plano, xy_um_por_pixel, xy_um_por_pixel)

gfap_umbral_percentil = int(datos.get("gfap_umbral_percentil", 85))
gfap_apertura_px      = int(datos.get("gfap_apertura_px", 1))   # reducido de 3 a 1
gfap_cierre_px        = int(datos.get("gfap_cierre_px", 2))     # reducido de 15 a 2
gfap_erosion_px       = int(datos.get("gfap_erosion_px", 0))    # eliminado erosión

print(f"Escala Z: {z_um_por_plano} µm/plano  |  XY: {xy_um_por_pixel:.4f} µm/píxel")
print(f"Canales: {canales}")


# ============================================================
# UTILIDADES
# ============================================================
def asegurar_zyx(vol):
    """Pone el eje más largo en posición 0 (Z)."""
    idx_max = int(np.argmax(vol.shape))
    if idx_max != 0:
        orden = [idx_max] + [i for i in range(3) if i != idx_max]
        vol   = np.transpose(vol, orden)
        print(f"    Transpuesto a (Z,Y,X): {vol.shape}")
    return vol


def a_float32(vol):
    """
    Convierte cualquier dtype a float32 normalizado en [0, 1].
    Vispy acepta float32 de forma universal, independientemente
    de la versión de napari instalada.
    """
    vol = vol.astype(np.float32)
    mn, mx = vol.min(), vol.max()
    if mx > mn:
        vol = (vol - mn) / (mx - mn)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)
    return np.ascontiguousarray(vol, dtype=np.float32)


def cargar_vol(clave):
    """Carga un volumen del npz, lo normaliza a float32 [0,1] y asegura (Z,Y,X)."""
    if clave not in datos:
        return None
    vol = np.squeeze(datos[clave])
    if vol.ndim != 3:
        print(f"  [AVISO] {clave} shape={vol.shape} no es 3D, se omite")
        return None
    vol = asegurar_zyx(vol)
    vol = a_float32(vol)
    return vol


def calcular_contraste(vol, p_low=P_LOW, p_high=P_HIGH):
    """Percentiles sobre float32 [0,1]."""
    pos = vol[vol > 0]
    if pos.size == 0:
        return [0.0, 1.0]
    lo = float(np.percentile(pos, p_low))
    hi = float(np.percentile(pos, p_high))
    hi = max(hi, lo + 1e-4)
    return [max(0.0, lo), min(1.0, hi)]


def crear_capa_contacto(vol_iba1, vol_gfap, sigma=2.0, umbral_percentil=80):
    """
    Capa amarilla de co-localización IBA-1 x GFAP.
    Ambos volúmenes ya vienen en float32 [0,1].
    """
    iba_s  = ndimage.gaussian_filter(vol_iba1.astype(np.float32), sigma=sigma)
    gfap_s = ndimage.gaussian_filter(vol_gfap.astype(np.float32), sigma=sigma)
    contacto = iba_s * gfap_s
    pos = contacto[contacto > 0]
    if pos.size > 0:
        umbral = float(np.percentile(pos, umbral_percentil))
        contacto[contacto < umbral] = 0
    mx = contacto.max()
    if mx > 0:
        contacto = contacto / mx
    return np.ascontiguousarray(contacto, dtype=np.float32)


def segmentar_gfap_live(vol_gfap_f32, umbral_percentil, erosion_px,
                         apertura_px, cierre_px):
    """
    Recalcula máscara GFAP sobre el volumen float32 [0,1].
    Devuelve float32 [0,1] listo para napari.
    """
    pos = vol_gfap_f32[vol_gfap_f32 > 0]
    if pos.size == 0:
        return np.zeros_like(vol_gfap_f32)

    umbral  = float(np.percentile(pos, umbral_percentil))
    mascara = (vol_gfap_f32 >= umbral).astype(np.uint8)

    struct = ndimage.generate_binary_structure(3, 1)

    if erosion_px > 0:
        mascara = ndimage.binary_erosion(
            mascara, structure=struct, iterations=erosion_px).astype(np.uint8)
    if apertura_px > 0:
        mascara = ndimage.binary_opening(
            mascara, structure=struct, iterations=apertura_px).astype(np.uint8)
    if cierre_px > 0:
        mascara = ndimage.binary_closing(
            mascara, structure=struct,
            iterations=min(2, cierre_px)).astype(np.uint8)

    etiquetas, n_comp = ndimage.label(mascara)
    if n_comp > 1:
        tamanhos = ndimage.sum(mascara, etiquetas, range(1, n_comp + 1))
        mayor    = int(np.argmax(tamanhos)) + 1
        mascara  = (etiquetas == mayor).astype(np.uint8)

    resultado = vol_gfap_f32 * mascara.astype(np.float32)
    return np.ascontiguousarray(resultado, dtype=np.float32)


# ============================================================
# 2. CARGAR VOLÚMENES
# ============================================================
vol_dapi  = cargar_vol("vol_C00_DAPI")
vol_iba1  = cargar_vol("vol_C01_IBA_1")
vol_gfap  = cargar_vol("vol_C02_GFAP")
vol_tumor = cargar_vol("vol_C02_GFAP_mascara")

if vol_tumor is None and vol_gfap is not None:
    print("  Generando mascara GFAP desde el volumen completo...")
    vol_tumor = segmentar_gfap_live(
        vol_gfap, gfap_umbral_percentil, gfap_erosion_px,
        gfap_apertura_px, gfap_cierre_px)

# ============================================================
# 3. VISUALIZACIÓN EN NAPARI
# ============================================================
viewer = napari.Viewer(ndisplay=3)
capas  = {}

if vol_dapi is not None:
    cl = calcular_contraste(vol_dapi)
    print(f"  DAPI       shape={vol_dapi.shape}  cl={[round(v,4) for v in cl]}")
    capas["DAPI"] = viewer.add_image(
        vol_dapi, name="DAPI", scale=scale_xyz,
        blending="additive", colormap="blue",
        contrast_limits=cl, opacity=0.35, gamma=1.0,
    )

if vol_iba1 is not None:
    # Percentil alto más agresivo para IBA-1: solo muestra microglía brillante,
    # suprime el ruido de fondo disperso
    cl = calcular_contraste(vol_iba1, p_low=60, p_high=99)
    print(f"  IBA-1      shape={vol_iba1.shape}  cl={[round(v,4) for v in cl]}")
    capas["IBA-1"] = viewer.add_image(
        vol_iba1, name="IBA-1", scale=scale_xyz,
        blending="additive", colormap="green",
        contrast_limits=cl, opacity=0.7, gamma=1.0,
    )

# Capa GFAP-fondo eliminada — solo se muestra el tumor segmentado

if vol_tumor is not None:
    print(f"  GFAP-tumor shape={vol_tumor.shape}")

    capas["GFAP-tumor"] = viewer.add_labels(
        vol_tumor.astype(np.uint8),
        name="GFAP-tumor",
        scale=scale_xyz
    )

if vol_iba1 is not None and vol_tumor is not None:
    print("  Calculando capa de contacto IBA-1 x GFAP...")

    # 🔹 1. Binarizar IBA-1 (microglía activa)
    umbral_iba = np.percentile(vol_iba1[vol_iba1 > 0], 80)
    mask_iba   = (vol_iba1 >= umbral_iba).astype(np.uint8)

    # 🔹 2. GFAP ya es máscara
    mask_gfap = (vol_tumor > 0).astype(np.uint8)

    # 🔹 3. Intersección (contacto real)
    vol_contacto = (mask_iba & mask_gfap).astype(np.uint8)

    print(f"  Contacto voxels: {vol_contacto.sum():,}")

    capas["Contacto"] = viewer.add_labels(
        vol_contacto,
        name="Contacto IBA1xGFAP",
        scale=scale_xyz
    )

# Ángulo de cámara optimizado para ver el tumor de frente
# El volumen es alargado en Z (601 planos) vs XY (388x387 px)
# Rotamos para ver la masa tumoral desde un ángulo que muestre su volumen real
viewer.camera.angles = (0, 30, 135)  # vista lateral-frontal del tumor
viewer.camera.zoom   = 1.2

# ============================================================
# 4. WIDGET: REFINAR MASCARA GFAP EN TIEMPO REAL
# ============================================================
@magicgui(
    call_button="Aplicar mascara",
    umbral_percentil    ={"widget_type": "SpinBox", "min": 50, "max": 99, "value": gfap_umbral_percentil, "label": "Umbral (percentil)"},
    erosion_px          ={"widget_type": "SpinBox", "min": 0,  "max": 10, "value": gfap_erosion_px,       "label": "Erosion (px)"},
    apertura_px         ={"widget_type": "SpinBox", "min": 0,  "max": 20, "value": gfap_apertura_px,      "label": "Apertura (px)"},
    cierre_px           ={"widget_type": "SpinBox", "min": 0,  "max": 30, "value": gfap_cierre_px,        "label": "Cierre (px)"},
    recalcular_contacto ={"label": "Recalcular contacto"},
)
def widget_gfap(umbral_percentil=gfap_umbral_percentil,
                erosion_px=gfap_erosion_px,
                apertura_px=gfap_apertura_px,
                cierre_px=gfap_cierre_px,
                recalcular_contacto=True):
    if vol_gfap is None:
        print("[ERROR] Volumen GFAP no disponible.")
        return

    vol_mask = segmentar_gfap_live(
        vol_gfap, umbral_percentil, erosion_px, apertura_px, cierre_px)

    if "GFAP-tumor" in capas:
        capas["GFAP-tumor"].data            = vol_mask
        capas["GFAP-tumor"].contrast_limits = calcular_contraste(vol_mask)

    if recalcular_contacto and "Contacto" in capas and vol_iba1 is not None:
        vol_c = crear_capa_contacto(vol_iba1, vol_mask)
        capas["Contacto"].data            = vol_c
        capas["Contacto"].contrast_limits = calcular_contraste(vol_c)

    n = int((vol_mask > 0).sum())
    t = int((vol_gfap  > 0).sum())
    print(f"Mascara: {n:,}/{t:,} voxeles ({100*n/max(t,1):.1f}%)")
    print(f"  GFAP_UMBRAL_PERCENTIL = {umbral_percentil}")
    print(f"  GFAP_EROSION_PX       = {erosion_px}")
    print(f"  GFAP_APERTURA_PX      = {apertura_px}")
    print(f"  GFAP_CIERRE_PX        = {cierre_px}")

viewer.window.add_dock_widget(widget_gfap, area="right", name="Mascara GFAP")

# ============================================================
# 5. WIDGET: CONTRASTE Y OPACIDAD GENERAL
# ============================================================
nombres_capas = list(capas.keys())

@magicgui(
    call_button="Aplicar",
    capa    ={"choices": nombres_capas,     "label": "Capa"},
    opacidad={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "value": 0.7, "label": "Opacidad"},
    p_low   ={"widget_type": "SpinBox",     "min": 0,   "max": 89,  "value": P_LOW,  "label": "Percentil bajo"},
    p_high  ={"widget_type": "SpinBox",     "min": 90,  "max": 100, "value": P_HIGH, "label": "Percentil alto"},
    visible ={"label": "Visible"},
)
def widget_general(capa=nombres_capas[0], opacidad=0.7,
                   p_low=P_LOW, p_high=P_HIGH, visible=True):
    if capa not in capas:
        return
    layer                 = capas[capa]
    layer.opacity         = opacidad
    layer.visible         = visible
    cl                    = calcular_contraste(layer.data, p_low, p_high)
    layer.contrast_limits = cl
    print(f"[{capa}] opacidad={opacidad:.2f}  contraste=[{cl[0]:.4f}, {cl[1]:.4f}]")

viewer.window.add_dock_widget(widget_general, area="right", name="General")

# ============================================================
# RESUMEN
# ============================================================
print("\n" + "=" * 60)
print("NAPARI LISTO")
print(f"  Escala Z:  {z_um_por_plano} µm/plano")
print(f"  Escala XY: {xy_um_por_pixel:.4f} µm/píxel")
print(f"  Capas: {list(capas.keys())}")
print("  Panel 'Mascara GFAP' -> refina el tumor en tiempo real")
print("  Capa 'Contacto IBA1xGFAP' -> infiltracion microglía (amarillo)")
print("  Capa 'GFAP-fondo' -> activala para ver contexto astrocítico")
print("=" * 60)

napari.run()