import os
import numpy as np
import napari
from skimage import io, transform, filters

# === CONFIGURACIÓN ===
pixel_size = 1.1364  
z_step = 40          
ruta_raiz = r"C:\Users\frans\Desktop\ct2\RECONSTRUCCION_V1" 
canales = ["C00-DAPI", "C01-IBA-1", "C02-GFAP"]
colores = ["blue", "green", "red"]
escala_reduccion = 0.5 

# 1. ENCONTRAR EL TAMAÑO MÁXIMO CON MARGEN DE SEGURIDAD 
max_h, max_w = 0, 0
print("Calculando dimensiones máximas...")
for canal in canales:
    ruta = os.path.join(ruta_raiz, canal)
    if os.path.exists(ruta):
        for f in os.listdir(ruta):
            if f.endswith('.tif'):
                img_shape = io.imread(os.path.join(ruta, f)).shape
                max_h = max(max_h, int(img_shape[0] * escala_reduccion) + 2)
                max_w = max(max_w, int(img_shape[1] * escala_reduccion) + 2)

viewer = napari.Viewer(ndisplay=3)

for canal, color in zip(canales, colores):
    ruta_canal = os.path.join(ruta_raiz, canal)
    if not os.path.exists(ruta_canal): continue
    
    archivos = sorted([f for f in os.listdir(ruta_canal) if f.endswith('.tif')])
    stack_canal = []
    
    print(f"Procesando {canal}...")
    for f in archivos:
        img = io.imread(os.path.join(ruta_canal, f))
        
        # Redimensionar
        img_res = transform.rescale(img, escala_reduccion, preserve_range=True).astype(np.uint16)
        h, w = img_res.shape
        
        # CREAR EL MARCO (Lienzo negro)
        canvas = np.zeros((max_h, max_w), dtype=np.uint16)
        
        # AJUSTE DE SEGURIDAD: Solo pegamos lo que quepa en el canvas 
        # (por si acaso h o w son mayores por 1 píxel que max_h o max_w)
        limite_h = min(h, max_h)
        limite_w = min(w, max_w)
        
        canvas[:limite_h, :limite_w] = img_res[:limite_h, :limite_w]
        
        stack_canal.append(canvas)
    
    stack_array = np.stack(stack_canal)
    z_scale = (z_step / pixel_size) * (1 / escala_reduccion)

    # --- AJUSTE ESTRATÉGICO DE BRILLOS ---
    if canal == "C00-DAPI":
        # Bajamos el límite para que los núcleos azules brillen más
        limites = [0, 400] 
    
    elif canal == "C01-IBA-1":
        # El verde es tu canal estrella, vamos a darle potencia
        limites = [0, 300] 
        
    elif canal == "C02-GFAP":
        # El rojo suele ser el más débil. Bajamos mucho el límite para 
        # que hasta la señal más floja se vea roja brillante.
        # Si sigue tapando mucho, sube este 150 a 300.
        limites = [0, 150] 

    # Aplicamos la visualización
    viewer.add_image(
        stack_array,
        name=canal,
        scale=(z_scale, 1, 1),
        blending="additive", # Crucial: esto suma los colores en lugar de taparlos
        colormap=color,
        contrast_limits=limites,
        opacity=0.7 if canal == "C02-GFAP" else 1.0 # El rojo un poco transparente
    )
   
# Datos de tu experimento
#indices = [46, 48, 49, 52, 54, 56, 58, 60, 62, 64, 67, 68, 70, 71, 73]

pixel_size = 1.1364
grosor_corte = 20 # um

# Supongamos que ya tenemos las áreas en píxeles (esto es un ejemplo)
# areas_pixeles = [valor1, valor2, ..., valor14]

def calcular_volumen_final(areas_pixeles):
    volumen_total_um3 = 0
    
    for i in range(len(indices) - 1):
        # 1. Áreas en micras cuadradas
        a1 = areas_pixeles[i] * (pixel_size**2)
        a2 = areas_pixeles[i+1] * (pixel_size**2)
        
        # 2. Distancia entre estos dos cortes
        distancia_z = (indices[i+1] - indices[i]) * grosor_corte
        
        # 3. Volumen del trapezoide entre cortes (Cavalieri refinado)
        volumen_tramo = ((a1 + a2) / 2) * distancia_z
        volumen_total_um3 += volumen_tramo
        
    # Pasar a mm3
    return volumen_total_um3 / 1e9

print(f"Volumen tumoral: {calcular_volumen_final(areas_pixeles)} mm3")

print("¡Hecho! Napari debería abrirse sin errores de tamaño ahora.")
napari.run()