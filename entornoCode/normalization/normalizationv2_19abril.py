import numpy as np
import napari
from napari.qt.threading import thread_worker
from magicgui import magicgui
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN
# ============================================================
archivo_entrada = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\volumenes\volumenes.npz"

# ============================================================
# 1. CARGAR DATOS
# ============================================================
print(f"Cargando volúmenes desde: {archivo_entrada}")
datos = np.load(archivo_entrada, allow_pickle=False)

pixel_size       = float(datos["pixel_size"])
escala_reduccion = float(datos["escala_reduccion"])
canales          = list(datos["canales"])
colores          = list(datos["colores"])

print(f"Canales encontrados: {canales}")
print(f"pixel_size={pixel_size} µm  |  escala_reduccion={escala_reduccion}")

# ============================================================
# 2. CORRECCIÓN DE ESCALA
#
# El volumen tiene 1 plano por µm en Z (interpolado cada 1 µm,
# aunque los cortes reales disten 40 µm).
# Napari necesita saber cuántos µm reales ocupa cada voxel.
#
#   Z: la interpolación generó 1 plano por µm → z_um_por_plano = 1.0
#      (si cambias micras_entre_indices en el script de normalización,
#       ajusta también este valor)
#
#   XY: al reducir la imagen con escala_reduccion, cada píxel del array
#       agrupa 1/escala_reduccion píxeles originales, por lo que su
#       tamaño físico aumenta:
#       xy_um_por_pixel = pixel_size / escala_reduccion
# ============================================================
xy_um_por_pixel = pixel_size / escala_reduccion   # µm por píxel en el array reducido
z_um_por_plano  = 1.0                              # 1 plano = 1 µm (interpolado)

scale_xyz = (z_um_por_plano, xy_um_por_pixel, xy_um_por_pixel)
print(f"\nEscala física corregida:")
print(f"  Z  = {z_um_por_plano} µm/plano")
print(f"  XY = {xy_um_por_pixel:.4f} µm/píxel")

# ============================================================
# 3. CONTRASTE AUTOMÁTICO POR CANAL
#
# En lugar de usar [0, 300] fijo, calculamos percentiles reales
# del volumen para ajustar el contraste a los datos.
#
#   p_low  = percentil inferior → suprime el fondo/ruido
#   p_high = percentil superior → evita que un píxel brillante
#            sature toda la escala
# ============================================================
P_LOW  = 1    # percentil inferior (1% → elimina ruido de fondo)
P_HIGH = 99   # percentil superior (99% → ignora saturaciones)

def calcular_contraste(volume_3d, p_low=P_LOW, p_high=P_HIGH):
    """Calcula contrast_limits robustos ignorando píxeles negros (fondo)."""
    datos_validos = volume_3d[volume_3d > 0]
    if datos_validos.size == 0:
        return [0, 1]
    low  = float(np.percentile(datos_validos, p_low))
    high = float(np.percentile(datos_validos, p_high))
    # Asegura que haya al menos un rango mínimo para evitar errores en napari
    if high <= low:
        high = low + 1
    return [low, high]

# ============================================================
# 4. VISUALIZACIÓN EN NAPARI
# ============================================================
viewer = napari.Viewer(ndisplay=3)

capas = {}  # guardamos referencia a cada capa para el widget

# Opacidades y gammas por canal (ajusta a tu gusto)
config_canales = {
    "C00-DAPI":  {"opacity": 0.5,  "gamma": 1.0, "visible": True},
    "C01-IBA-1": {"opacity": 0.8,  "gamma": 1.2, "visible": True},
    "C02-GFAP":  {"opacity": 1.0,  "gamma": 0.8, "visible": True},
}
config_default = {"opacity": 0.7, "gamma": 1.0, "visible": True}

for i, (canal, color) in enumerate(zip(canales, colores)):
    clave = f"vol_{canal.replace('-', '_')}"
    if clave not in datos:
        print(f"[AVISO] No se encontró el volumen para: {canal}")
        continue

    volume_3d = datos[clave]
    cl        = calcular_contraste(volume_3d)
    cfg       = config_canales.get(canal, config_default)

    print(f"\nCanal '{canal}'")
    print(f"  Shape:    {volume_3d.shape}")
    print(f"  Contraste automático: [{cl[0]:.1f}, {cl[1]:.1f}]")
    print(f"  Min/Max real:         [{volume_3d.min()}, {volume_3d.max()}]")

    capa = viewer.add_image(
        volume_3d,
        name=canal,
        scale=scale_xyz,
        blending="additive",
        colormap=color,
        contrast_limits=cl,
        opacity=cfg["opacity"],
        gamma=cfg["gamma"],
        visible=cfg["visible"],
    )
    capas[canal] = capa

# ============================================================
# 5. AJUSTE DE CÁMARA INICIAL
#
# Centra la cámara sobre el volumen y define un ángulo de vista
# que muestre la estructura tumoral de frente (no en diagonal).
# ============================================================
viewer.camera.angles = (0, 45, 45)   # (roll, pitch, yaw) en grados
viewer.camera.zoom   = 1.0

# ============================================================
# 6. WIDGET INTERACTIVO DE CONTRASTE
#
# Permite reajustar el contraste de cada canal en tiempo real
# sin salir de Napari.
# ============================================================
@magicgui(
    call_button="Aplicar contraste",
    canal={"choices": canales, "label": "Canal"},
    p_low={"widget_type": "SpinBox", "min": 0, "max": 49,  "label": "Percentil bajo"},
    p_high={"widget_type": "SpinBox", "min": 50, "max": 100, "label": "Percentil alto"},
)
def widget_contraste(canal=canales[0], p_low=P_LOW, p_high=P_HIGH):
    """Recalcula el contraste del canal seleccionado con los percentiles dados."""
    clave = f"vol_{canal.replace('-', '_')}"
    if clave not in datos:
        return
    vol = datos[clave]
    cl  = calcular_contraste(vol, p_low, p_high)
    if canal in capas:
        capas[canal].contrast_limits = cl
        print(f"[{canal}] Contraste actualizado: [{cl[0]:.1f}, {cl[1]:.1f}]")

viewer.window.add_dock_widget(widget_contraste, area="right", name="Contraste")

# ============================================================
# 7. WIDGET DE VISIBILIDAD Y OPACIDAD
# ============================================================
@magicgui(
    call_button="Aplicar",
    canal={"choices": canales, "label": "Canal"},
    opacidad={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0, "label": "Opacidad"},
    visible={"label": "Visible"},
)
def widget_opacidad(canal=canales[0], opacidad=0.7, visible=True):
    if canal in capas:
        capas[canal].opacity = opacidad
        capas[canal].visible = visible

viewer.window.add_dock_widget(widget_opacidad, area="right", name="Opacidad")

# ============================================================
# RESUMEN FINAL
# ============================================================
print("\n" + "="*50)
print("✓ Visualización lista.")
print(f"  Escala Z:  {z_um_por_plano} µm/plano")
print(f"  Escala XY: {xy_um_por_pixel:.4f} µm/píxel")
print("  Contraste: automático por percentiles")
print("  Widgets:   Contraste | Opacidad (panel derecho)")
print("="*50)

napari.run()