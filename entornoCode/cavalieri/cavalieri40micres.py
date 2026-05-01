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

# LISTAS BALANCEADAS (He unificado los nombres aquí para que coincidan con el bucle)
MUESTREOS_40 = {
    "Set A": [
        45, 47, 49, 
        51, # Compensación del 50
        54, # Compensación del 53 y 55
        57, 60, 63, 67, 69, 71, 73
    ],
    "Set B": [
        46, 48, 
        52, # Compensación del 53
        56, # Compensación del 55
        58, 62, 64, 68, 70
    ]
}

CANALES = ["C01-IBA-1", "C00-DAPI"]

# ============================================================
# FUNCIÓN DE CÁLCULO
# ============================================================
def calcular_volumen(canal, lista_indices, T_distancia):
    ruta_canal = os.path.join(ruta_raiz, canal)
    if not os.path.exists(ruta_canal):
        print(f"Error: No existe la ruta {ruta_canal}")
        return 0, []

    archivos = os.listdir(ruta_canal)
    areas_locales = []
    
    for n in lista_indices:
        match = [f for f in archivos if f.lower().startswith(f"c{n}") or f.startswith(str(n))]
        if match:
            try:
                img = io.imread(os.path.join(ruta_canal, match[0]))
                img_res = transform.rescale(img, escala_reduccion, preserve_range=True).astype(np.uint16)
                
                if "DAPI" in canal:
                    umbral = filters.threshold_otsu(img_res)
                else:
                    umbral = filters.threshold_li(img_res)
                
                mascara = img_res > umbral
                mascara = morphology.remove_small_objects(mascara, min_size=50)
                
                area = np.sum(mascara) * area_pixel_micras2
                areas_locales.append(area)
            except:
                areas_locales.append(0)
    
    volumen_total = sum(areas_locales) * T_distancia
    return volumen_total, areas_locales

# ============================================================
# EJECUCIÓN (Corregida la llamada a los nombres de las listas)
# ============================================================
print("--- INICIANDO VALIDACIÓN CRUZADA DE VOLUMEN (T=40µm) ---")

for canal in CANALES:
    print(f"\n" + "="*50)
    print(f"CANAL: {canal}")
    print("="*50)
    
    # IMPORTANTE: Ahora los nombres coinciden con el diccionario MUESTREOS_40
    v_a, _ = calcular_volumen(canal, MUESTREOS_40["Set A"], 40)
    v_b, _ = calcular_volumen(canal, MUESTREOS_40["Set B"], 40)
    
    print(f"Volumen Set A: {v_a/1e9:.6f} mm³")
    print(f"Volumen Set B: {v_b/1e9:.6f} mm³")
    
    if v_a > 0:
        error = abs(v_a - v_b) / v_a * 100
        print(f"Diferencia relativa (consistencia interna): {error:.2f}%")

print("\n✓ Proceso finalizado.")