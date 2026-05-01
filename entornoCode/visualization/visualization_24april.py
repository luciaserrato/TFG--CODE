import numpy as np
import napari
from magicgui import magicgui
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN
# ============================================================
archivo_entrada = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\volumenes\volumenes.npz"

# ============================================================
# 1. CARGAR DATOS Y METADATOS
# ============================================================
print(f"Cargando: {archivo_entrada}")
datos = np.load(archivo_entrada, allow_pickle=False)

canales  = list(datos["canales"])           # canales base: DAPI, IBA-1, GFAP
colores  = list(datos["colores"])

z_um_por_plano  = float(datos["z_um_por_plano"])
xy_um_por_pixel = float(datos["xy_um_por_pixel"])
scale_xyz       = (z_um_por_plano, xy_um_por_pixel, xy_um_por_pixel)

# Parámetros de máscara guardados por el normalizador
gfap_umbral_percentil = int(datos["gfap_umbral_percentil"])
gfap_apertura_px      = int(datos["gfap_apertura_px"])
gfap_cierre_px        = int(datos["gfap_cierre_px"])

# Contraste: shape (n_canales_guardados, 2)
cl_raw            = datos["contrast_limits"]
canales_guardados = list(datos["canales_guardados"])
contrast_map      = {}
for i, c in enumerate(canales_guardados):
    contrast_map[c] = cl_raw[i].tolist() if cl_raw.ndim == 2 else cl_raw.tolist()

print(f"Escala Z:  {z_um_por_plano} µm/plano  |  XY: {xy_um_por_pixel:.4f} µm/píxel")
print(f"Canales:   {canales}")

# ============================================================
# 2. VISUALIZACIÓN EN NAPARI
# ============================================================
viewer = napari.Viewer(ndisplay=3)

# Config visual por canal
config_canales = {
    "C00-DAPI":        {"color": "blue",  "opacity": 0.4, "gamma": 1.0},
    "C01-IBA-1":       {"color": "green", "opacity": 0.7, "gamma": 1.2},
    "C02-GFAP":        {"color": "red",   "opacity": 0.3, "gamma": 1.0},  # fondo atenuado
    "C02-GFAP_mascara":{"color": "red",   "opacity": 1.0, "gamma": 0.8},  # tumor destacado
}

capas = {}

# Canales base (DAPI e IBA-1)
for canal, color in zip(canales[:2], colores[:2]):
    clave = f"vol_{canal.replace('-', '_')}"
    if clave not in datos:
        print(f"[AVISO] No se encontró: {clave}")
        continue
    volume_3d = datos[clave]
    cfg       = config_canales.get(canal, {"color": color, "opacity": 0.7, "gamma": 1.0})
    cl        = contrast_map.get(canal, [0, 300])
    

    print(f"  '{canal}'  shape={volume_3d.shape}  contraste={[round(v,1) for v in cl]}")
    capa = viewer.add_image(
        volume_3d, name=canal, scale=scale_xyz,
        blending="additive", colormap=cfg["color"],
        contrast_limits=cl, opacity=cfg["opacity"], gamma=cfg["gamma"],
    )
    capas[canal] = capa

# GFAP original (fondo, atenuado) — permite ver el contexto astrocítico completo
canal_gfap      = "C02-GFAP"
clave_gfap      = "vol_C02_GFAP"
if clave_gfap in datos:
    vol_gfap = datos[clave_gfap]
    cl_gfap  = contrast_map.get(canal_gfap, [0, 300])
    cfg      = config_canales[canal_gfap]
    print(f"  '{canal_gfap}' (fondo)  shape={vol_gfap.shape}")
    capa = viewer.add_image(
        vol_gfap, name="GFAP-fondo", scale=scale_xyz,
        blending="additive", colormap=cfg["color"],
        contrast_limits=cl_gfap, opacity=cfg["opacity"], gamma=cfg["gamma"],
        visible=False,   # oculto por defecto; activar desde el panel de capas
    )
    capas["GFAP-fondo"] = capa

# GFAP enmascarado (tumor) — capa principal activa
clave_mascara = "vol_C02_GFAP_mascara"
if clave_mascara in datos:
    vol_mask = datos[clave_mascara]
    cl_mask  = contrast_map.get("C02-GFAP_mascara", [0, 300])
    cfg      = config_canales["C02-GFAP_mascara"]
    print(f"  'GFAP-tumor'  shape={vol_mask.shape}  contraste={[round(v,1) for v in cl_mask]}")
    capa = viewer.add_image(
        vol_mask, name="GFAP-tumor", scale=scale_xyz,
        blending="additive", colormap=cfg["color"],
        contrast_limits=cl_mask, opacity=cfg["opacity"], gamma=cfg["gamma"],
    )
    capas["GFAP-tumor"] = capa

viewer.camera.angles = (0, 45, 45)
viewer.camera.zoom   = 1.0

# ============================================================
# 3. WIDGET: UMBRAL GFAP EN TIEMPO REAL
#
# Permite regenerar la máscara sin salir de Napari ni reejecutar
# el normalizador. Útil para encontrar el umbral óptimo.
# Una vez satisfecho con el resultado, copia los valores al
# normalizador y regenera el npz para guardarlos permanentemente.
# ============================================================
@magicgui(
    call_button="Aplicar máscara",
    umbral_percentil={"widget_type": "SpinBox",   "min": 50,  "max": 99,  "value": gfap_umbral_percentil, "label": "Umbral percentil"},
    apertura_px=     {"widget_type": "SpinBox",   "min": 0,   "max": 20,  "value": gfap_apertura_px,      "label": "Apertura (px)"},
    cierre_px=       {"widget_type": "SpinBox",   "min": 0,   "max": 30,  "value": gfap_cierre_px,        "label": "Cierre (px)"},
    p_contraste_low= {"widget_type": "SpinBox",   "min": 0,   "max": 49,  "value": 1,                     "label": "Contraste bajo (%)"},
    p_contraste_high={"widget_type": "SpinBox",   "min": 50,  "max": 100, "value": 99,                    "label": "Contraste alto (%)"},
)
def widget_gfap(umbral_percentil=gfap_umbral_percentil,
                apertura_px=gfap_apertura_px,
                cierre_px=gfap_cierre_px,
                p_contraste_low=1,
                p_contraste_high=99):
    """
    Recalcula la máscara GFAP con los parámetros elegidos y actualiza
    la capa 'GFAP-tumor' en tiempo real.

    umbral_percentil : sube para aislar solo el núcleo más brillante
    apertura_px      : elimina puntos aislados fuera del tumor
    cierre_px        : rellena huecos dentro de la masa tumoral
    """
    if clave_gfap not in datos:
        print("[ERROR] No se encontró el volumen GFAP en el npz.")
        return

    vol = datos[clave_gfap]

    # Umbral
    validos = vol[vol > 0]
    if validos.size == 0:
        return
    umbral  = float(np.percentile(validos, umbral_percentil))
    mascara = (vol >= umbral).astype(np.uint8)

    # Morfología
    struct = ndimage.generate_binary_structure(3, 1)
    if apertura_px > 0:
        mascara = ndimage.binary_opening(mascara, structure=struct,
                                         iterations=apertura_px).astype(np.uint8)
    if cierre_px > 0:
        mascara = ndimage.binary_closing(mascara, structure=struct,
                                         iterations=cierre_px).astype(np.uint8)

    vol_enmascarado = (vol * mascara).astype(np.uint16)

    # Contraste sobre píxeles conservados
    conservados = vol_enmascarado[vol_enmascarado > 0]
    if conservados.size > 0:
        lo = float(np.percentile(conservados, p_contraste_low))
        hi = float(np.percentile(conservados, p_contraste_high))
        hi = max(hi, lo + 1)
    else:
        lo, hi = 0.0, 1.0

    # Actualizar capa
    if "GFAP-tumor" in capas:
        capas["GFAP-tumor"].data             = vol_enmascarado
        capas["GFAP-tumor"].contrast_limits  = [lo, hi]

    n = int(mascara.sum())
    t = int((vol > 0).sum())
    print(f"Máscara actualizada → umbral={umbral:.0f}  "
          f"vóxeles={n:,}/{t:,} ({100*n/max(t,1):.1f}%)  "
          f"contraste=[{lo:.0f}, {hi:.0f}]")
    print(f"  → Copia estos valores al normalizador para guardar permanentemente:")
    print(f"     GFAP_UMBRAL_PERCENTIL = {umbral_percentil}")
    print(f"     GFAP_APERTURA_PX      = {apertura_px}")
    print(f"     GFAP_CIERRE_PX        = {cierre_px}")

viewer.window.add_dock_widget(widget_gfap, area="right", name="Máscara GFAP")

# ============================================================
# 4. WIDGET: CONTRASTE Y OPACIDAD GENERALES
# ============================================================
todos_los_nombres = list(capas.keys())

@magicgui(
    call_button="Aplicar",
    capa=    {"choices": todos_los_nombres, "label": "Capa"},
    opacidad={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "value": 0.7, "label": "Opacidad"},
    p_low=   {"widget_type": "SpinBox", "min": 0,  "max": 49,  "value": 1,  "label": "Percentil bajo"},
    p_high=  {"widget_type": "SpinBox", "min": 50, "max": 100, "value": 99, "label": "Percentil alto"},
    visible= {"label": "Visible"},
)
def widget_general(capa=todos_los_nombres[0], opacidad=0.7,
                   p_low=1, p_high=99, visible=True):
    if capa not in capas:
        return
    layer = capas[capa]
    layer.opacity = opacidad
    layer.visible = visible

    vol = layer.data
    validos = vol[vol > 0]
    if validos.size > 0:
        lo = float(np.percentile(validos, p_low))
        hi = float(np.percentile(validos, p_high))
        layer.contrast_limits = [lo, max(hi, lo + 1)]
        print(f"[{capa}] contraste=[{lo:.1f}, {hi:.1f}]  opacidad={opacidad:.2f}")

viewer.window.add_dock_widget(widget_general, area="right", name="General")

print("\n✓ Napari listo.")
print("  Panel derecho → 'Máscara GFAP': ajusta umbral, apertura y cierre en tiempo real.")
print("  Panel derecho → 'General':      ajusta contraste y opacidad de cualquier capa.")
print("  Capa 'GFAP-fondo' está oculta; actívala desde el panel de capas para ver contexto.")
napari.run()