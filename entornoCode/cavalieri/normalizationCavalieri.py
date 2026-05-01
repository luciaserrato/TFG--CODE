import os
import numpy as np
from skimage import io, transform, filters, morphology

# ============================================================
# CONFIGURACIÓN
# ============================================================
pixel_size           = 1.1364  
escala_reduccion     = 0.2
pixel_size_rescaled  = pixel_size / escala_reduccion 
area_pixel_micras2   = pixel_size_rescaled ** 2

ruta_raiz = r"C:\Users\frans\Desktop\TFG\ct2\Reconstruccion_v2"
#canal_referencia = "C00-DAPI" 
canal_referencia = "C01-IBA-1" 
#canal_referencia = "C02-GFAP" 

# Definimos los muestreos
muestreos = {
    "Cavalieri 40um(impares)": {
        # Usamos saltos de 2, pero solo si el archivo existe
        "indices": [45, 47, 49, 51, 53, 55, 57, 60, 61, 63, 65, 67, 69, 71, 73], 
        "T": 40 
    },
    "Cavalieri 40um(pares)": {
        # Usamos saltos de 2, pero solo si el archivo existe
        "indices": [46, 48, 49, 52, 54, 56, 58, 60, 62, 64, 67, 68, 70, 71, 73], 
        "T": 40 
    },
    "Cavalieri 60um (Multiplos de 3 disponibles)": {
        # Usamos saltos de 3, pero solo si el archivo existe, el 65 pq 66 no está
        "indices": [45, 48, 51, 54, 57, 60, 63, 65, 69, 73], 
        "T": 60
    }
}

# ============================================================
# FUNCIÓN DE CÁLCULO (IMPORTANTE: Las variables aquí dentro son locales)
# ============================================================
def calcular_volumen_cavalieri(lista_indices, T_distancia):
    ruta_canal = os.path.join(ruta_raiz, canal_referencia)
    if not os.path.exists(ruta_canal):
        print(f"Error: No existe la ruta {ruta_canal}")
        return 0, []

    archivos_carpeta = os.listdir(ruta_canal)
    areas_locales = [] # Esta variable solo existe aquí dentro
    
    for n in lista_indices:
        match = [f for f in archivos_carpeta if f.lower().startswith(f"c{n}") or f.startswith(str(n))]
        if match:
            try:
                img = io.imread(os.path.join(ruta_canal, match[0]))
                img_res = transform.rescale(img, escala_reduccion, preserve_range=True).astype(np.uint16)
                
                # Usamos Li para IBA-1
                umbral = filters.threshold_li(img_res)
                mascara = img_res > umbral
                mascara = morphology.remove_small_objects(mascara, min_size=50)
                
                area_micras2 = np.sum(mascara) * area_pixel_micras2
                areas_locales.append(area_micras2)
            except Exception as e:
                areas_locales.append(0)
        else:
            print(f"  Aviso: No se encontró el corte {n}")

    volumen_total = sum(areas_locales) * T_distancia
    return volumen_total, areas_locales

# ============================================================
# EJECUCIÓN 
# ============================================================
resultados_finales = {}

print(f"Analizando canal IBA-1 para estimación de volumen...\n")
for nombre, config in muestreos.items():
    # LLAMADA A LA FUNCIÓN: guardamos los resultados en variables nuevas
    vol, lista_de_areas = calcular_volumen_cavalieri(config["indices"], config["T"])
    
    resultados_finales[nombre] = vol
    
    print(f"--- {nombre} ---")
    print(f"  Cortes encontrados: {len(lista_de_areas)}")
    print(f"  Sumatorio de áreas: {sum(lista_de_areas):.2f} µm²")
    print(f"  Volumen estimado: {vol/1e9:.6f} mm³\n")

# --- COMPARATIVA FINAL ---
if "Cavalieri 40µm (Cada 2 cortes)" in resultados_finales:
    v40 = resultados_finales["Cavalieri 40µm (Cada 2 cortes)"]
    v60 = resultados_finales["Cavalieri 60µm (Cada 3 cortes)"]
    if v40 > 0:
        error = abs(v40 - v60) / v40 * 100
        print(f"Diferencia relativa entre muestreos: {error:.2f}%")