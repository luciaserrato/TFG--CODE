import os
import numpy as np
import napari
from skimage import io, transform, filters, morphology

# ============================================================
# CONFIGURACIÓN
# ============================================================
pixel_size        = 1.1364   # µm/píxel (XY)
z_step            = 40       # µm entre cortes (distancia real entre fotos)
ruta_raiz         = r"C:\Users\frans\Desktop\TFG\ct2\Reconstruccion_v2"

canales           = ["C00-DAPI", "C01-IBA-1", "C02-GFAP"]
colores           = ["blue",    "green",      "red"]

escala_reduccion  = 0.5
pasos_interpolacion = 8  

# ============================================================
# FUNCIONES TÉCNICAS
# ============================================================
def interpolar_stack(stack, pasos):
    if pasos <= 0: return stack
    Z, H, W = stack.shape
    Z_nuevo = (Z - 1) * (pasos + 1) + 1
    resultado = np.empty((Z_nuevo, H, W), dtype=np.uint16)
    for i in range(Z - 1):
        idx = i * (pasos + 1)
        A = stack[i]
        B = stack[i + 1]
        resultado[idx] = A
        for p in range(1, pasos + 1):
            t = p / (pasos + 1)
            resultado[idx + p] = (A * (1 - t) + B * t).astype(np.uint16)
    resultado[-1] = stack[-1]
    return resultado

# ============================================================
# PROCESO DE CARGA Y SEGMENTACIÓN
# ============================================================
viewer = napari.Viewer(ndisplay=3)

# Diccionario para guardar los stacks y poder cruzarlos (hacer la máscara)
stacks_finales = {}

# 1. Detectar tamaño máximo
max_h, max_w = 0, 0
for canal in canales:
    ruta = os.path.join(ruta_raiz, canal)
    if os.path.exists(ruta):
        archivos = [f for f in os.listdir(ruta) if f.endswith(".tif")]
        if archivos:
            img_test = io.imread(os.path.join(ruta, archivos[0]))
            h, w = img_test.shape
            max_h = max(max_h, int(h * escala_reduccion))
            max_w = max(max_w, int(w * escala_reduccion))

# 2. Cargar canales
for canal, color in zip(canales, colores):
    ruta_canal = os.path.join(ruta_raiz, canal)
    if not os.path.exists(ruta_canal): continue

    archivos = sorted([f for f in os.listdir(ruta_canal) if f.endswith(".tif")])
    if not archivos: continue

    print(f"Cargando {canal}...")
    stack_list = []
    for f in archivos:
        img = io.imread(os.path.join(ruta_canal, f))
        img_res = transform.resize(img, (max_h, max_w), preserve_range=True).astype(np.uint16)
        stack_list.append(img_res)

    stack_array = np.stack(stack_list)
    stacks_finales[canal] = stack_array

# ============================================================
# 🎭 CREACIÓN DE MÁSCARA Y FILTRADO DE GFAP
# ============================================================
print("Generando máscara tumoral para limpiar el canal GFAP...")

# Usamos IBA-1 para crear la máscara porque delimita mejor el tumor
iba1_ref = stacks_finales["C01-IBA-1"]
# Suavizamos un poco para unir los puntos de la microglía en una masa
iba1_smooth = filters.gaussian(iba1_ref, sigma=1)
thresh = filters.threshold_otsu(iba1_smooth)
mascara_tumor = iba1_smooth > (thresh * 1.2) # Multiplicamos por 1.2 para ser más selectivos

# Aplicamos la máscara al GFAP: Donde no haya tumor (máscara=0), el GFAP será 0
stacks_finales["C02-GFAP"] = (stacks_finales["C02-GFAP"] * mascara_tumor).astype(np.uint16)

# ============================================================
# VISUALIZACIÓN FINAL
# ============================================================
for canal in canales:
    if canal not in stacks_finales: continue
    
    stack_array = stacks_finales[canal]
    color = colores[canales.index(canal)]
    
    # Auto-contraste para GFAP ya filtrado
    limites = [0, 1000]
    if canal == "C02-GFAP":
        p_max = np.percentile(stack_array, 99.9)
        limites = [0, max(80, p_max)]

    if pasos_interpolacion > 0:
        stack_array = interpolar_stack(stack_array, pasos_interpolacion)

    z_scale = (z_step / pixel_size) * (1.0 / escala_reduccion) / (pasos_interpolacion + 1)

    viewer.add_image(
        stack_array,
        name=canal,
        scale=(z_scale, 1, 1),
        blending="additive",
        colormap=color,
        contrast_limits=limites
    )

print("\n✓ Reconstrucción con máscara completada.")
napari.run()