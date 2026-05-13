"""
visualizar_tfg_5may.py
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
#carpeta donde se encuentran los volúmenes .npy generados por el script de preprocesado
ARCHIVO_ENTRADA = r""

P_LOW  = 1
P_HIGH = 99
DAPI_UMBRAL_TUMORAL = 80

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

print(f"Escala Z: {z_um_por_plano} µm/plano  |  XY: {xy_um_por_pixel:.4f} µm/píxel")
print(f"Canales: {canales}")


# ============================================================
# UTILIDADES
# ============================================================
def asegurar_zyx(vol):
    idx_max = int(np.argmax(vol.shape))
    if idx_max != 0:
        orden = [idx_max] + [i for i in range(3) if i != idx_max]
        vol   = np.transpose(vol, orden)
        print(f"    Transpuesto a (Z,Y,X): {vol.shape}")
    return vol


def a_float32(vol):
    vol = vol.astype(np.float32)
    mn, mx = vol.min(), vol.max()
    if mx > mn:
        vol = (vol - mn) / (mx - mn)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)
    return np.ascontiguousarray(vol, dtype=np.float32)


def cargar_vol(clave):
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
    pos = vol[vol > 0]
    if pos.size == 0:
        return [0.0, 1.0]
    lo = float(np.percentile(pos, p_low))
    hi = float(np.percentile(pos, p_high))
    hi = max(hi, lo + 1e-4)
    return [max(0.0, lo), min(1.0, hi)]


def crear_capa_contacto(vol_iba1, vol_gfap, sigma=2.0, umbral_percentil=70,
                         dilatacion_z_um=40):
    mascara_gfap = (vol_gfap > 0).astype(np.float32)
    if dilatacion_z_um > 0:
        sigma_z = dilatacion_z_um / 3.0
        mascara_dilatada = ndimage.gaussian_filter1d(
            mascara_gfap, sigma=sigma_z, axis=0)
        mascara_dilatada = np.clip(mascara_dilatada, 0, 1)
    else:
        mascara_dilatada = mascara_gfap

    gfap_ponderado = vol_gfap.astype(np.float32) + mascara_dilatada * 0.3
    iba_s  = ndimage.gaussian_filter(vol_iba1.astype(np.float32), sigma=sigma)
    gfap_s = ndimage.gaussian_filter(gfap_ponderado, sigma=sigma)
    contacto = iba_s * gfap_s

    pos = contacto[contacto > 0]
    if pos.size > 0:
        umbral = float(np.percentile(pos, umbral_percentil))
        contacto[contacto < umbral] = 0
    mx = contacto.max()
    if mx > 0:
        contacto = contacto / mx
    return np.ascontiguousarray(contacto, dtype=np.float32)




def crear_dapi_tumoral(vol_dapi, umbral_percentil_dapi=80):
    """
    umbral_percentil_dapi: percentil sobre píxeles positivos para el corte
                           (80 → solo el 20% más brillante del DAPI)
    """
    pos = vol_dapi[vol_dapi > 0]
    if pos.size == 0:
        return np.zeros_like(vol_dapi)

    umbral = float(np.percentile(pos, umbral_percentil_dapi))
    dapi_tumoral = vol_dapi.copy()
    dapi_tumoral[dapi_tumoral < umbral] = 0   # suprime fondo difuso

    # Normalizar a [0,1]
    mx = dapi_tumoral.max()
    if mx > 0:
        dapi_tumoral = dapi_tumoral / mx
    return np.ascontiguousarray(dapi_tumoral, dtype=np.float32)


def crear_contacto_iba1_dapi(vol_iba1, vol_dapi, umbral_percentil_dapi=80,
                              sigma=2.0, umbral_percentil_contacto=70):
    """
    Co-localización IBA-1 × DAPI tumoral (cyan).

    Pasos:
      1. Restringir DAPI a píxeles brillantes (zona tumoral)
      2. Suavizar ambas señales para capturar regiones de solapamiento
      3. Multiplicar: alto valor solo donde AMBAS señales coinciden
      4. Umbralizar para mostrar solo los contactos más significativos
    """
    # DAPI restringido a zona tumoral
    dapi_tumoral = crear_dapi_tumoral(vol_dapi, umbral_percentil_dapi)

    # Suavizado para capturar vecindad espacial
    iba_s  = ndimage.gaussian_filter(vol_iba1.astype(np.float32),  sigma=sigma)
    dapi_s = ndimage.gaussian_filter(dapi_tumoral.astype(np.float32), sigma=sigma)

    # Producto: alta señal donde microglía y núcleos tumorales coinciden
    contacto = iba_s * dapi_s

    pos = contacto[contacto > 0]
    if pos.size > 0:
        umbral = float(np.percentile(pos, umbral_percentil_contacto))
        contacto[contacto < umbral] = 0

    mx = contacto.max()
    if mx > 0:
        contacto = contacto / mx
    return np.ascontiguousarray(contacto, dtype=np.float32)

def segmentar_gfap_live(vol_gfap_f32, umbral_percentil, erosion_px,
                         apertura_px, cierre_px):
    pos = vol_gfap_f32[vol_gfap_f32 > 0]
    if pos.size == 0:
        return np.zeros_like(vol_gfap_f32)

    umbral  = float(np.percentile(pos, umbral_percentil))
    mascara = (vol_gfap_f32 >= umbral).astype(np.uint8)
    struct  = ndimage.generate_binary_structure(3, 1)

    if erosion_px > 0:
        mascara = ndimage.binary_erosion(
            mascara, structure=struct, iterations=erosion_px).astype(np.uint8)
    if apertura_px > 0:
        mascara = ndimage.binary_opening(
            mascara, structure=struct, iterations=apertura_px).astype(np.uint8)
    if cierre_px > 0:
        mascara = ndimage.binary_closing(
            mascara, structure=struct, iterations=cierre_px).astype(np.uint8)

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
    print("  Sin mascara pre-calculada — mostrando GFAP completo")
    vol_tumor = vol_gfap.copy()

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
        contrast_limits=cl, opacity=0.5, gamma=1.0,
    )

if vol_iba1 is not None:
    cl = calcular_contraste(vol_iba1, p_low=60, p_high=99)
    print(f"  IBA-1      shape={vol_iba1.shape}  cl={[round(v,4) for v in cl]}")
    capas["IBA-1"] = viewer.add_image(
        vol_iba1, name="IBA-1", scale=scale_xyz,
        blending="additive", colormap="green",
        contrast_limits=cl, opacity=0.8, gamma=1.0,
    )

if vol_tumor is not None:
    cl = calcular_contraste(vol_tumor, p_low=60, p_high=99)
    print(f"  GFAP shape={vol_tumor.shape}  cl={[round(v,4) for v in cl]}")
    capas["GFAP"] = viewer.add_image(
        vol_tumor, name="GFAP", scale=scale_xyz,
        blending="additive", colormap="red",
        contrast_limits=cl, opacity=0.9, gamma=0.8,
    )

if vol_iba1 is not None and vol_tumor is not None:
    print("  Calculando capa de contacto IBA-1 x GFAP...")
    vol_contacto = crear_capa_contacto(vol_iba1, vol_tumor)
    cl = calcular_contraste(vol_contacto)
    print(f"  Contacto   shape={vol_contacto.shape}  cl={[round(v,4) for v in cl]}")
    capas["Contacto IBA1xGFAP"] = viewer.add_image(
        vol_contacto, name="Contacto IBA1xGFAP", scale=scale_xyz,
        blending="additive", colormap="yellow",
        contrast_limits=cl, opacity=1.0, gamma=0.7,
    )

# ── Co-localización IBA-1 × DAPI tumoral (cyan) ──────────────
# Muestra donde la microglía contacta con núcleos tumorales
# (DAPI brillante = alta densidad nuclear = zona tumoral)
if vol_iba1 is not None and vol_dapi is not None:
    print(f"  Calculando contacto IBA-1 x DAPI tumoral (umbral DAPI p{DAPI_UMBRAL_TUMORAL})...")
    vol_contacto_dapi = crear_contacto_iba1_dapi(
        vol_iba1, vol_dapi,
        umbral_percentil_dapi=DAPI_UMBRAL_TUMORAL,
        sigma=2.0,
        umbral_percentil_contacto=80
    )
    cl_dapi = calcular_contraste(vol_contacto_dapi)
    print(f"  ContactoDAPI shape={vol_contacto_dapi.shape}  cl={[round(v,4) for v in cl_dapi]}")
    capas["Contacto IBA1XDAPI"] = viewer.add_image(
        vol_contacto_dapi, name="Contacto IBA1xDAPI", scale=scale_xyz,
        blending="additive", colormap="cyan",
        contrast_limits=cl_dapi, opacity=1.0, gamma=0.7,
    )

viewer.camera.angles = (0, 30, 135)
viewer.camera.zoom   = 1.2

# ============================================================
# 4. WIDGET: CONTRASTE Y OPACIDAD GENERAL
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
print("=" * 60)

napari.run()