import os
import numpy as np
from skimage import io

# 

canal_nombre = "IBA-1"  # C00- DAPI, C01-Iba-1, C02-GFAP
ruta_entrada = rf"C:\Users\frans\Desktop\TFG\ct2\fotos\fallos\C01-IBA-1" 
ruta_salida = rf"C:\Users\frans\Desktop\TFG\ct2\Reconstruccion_v2\RESULTADOS_MIP_{canal_nombre}"

if not os.path.exists(ruta_salida):
    os.makedirs(ruta_salida)

# 1. Leer archivos
archivos = sorted([f for f in os.listdir(ruta_entrada) if f.endswith('.tif')])

# 2. Agrupar por muestra 
grupos = {}
for f in archivos:
    prefijo = f.split('--')[0] # Extrae "c46m" del nombre "c46--Z00--C00.tif", 
    #imagenes con m si no tiene m quitar
    if prefijo not in grupos:
        grupos[prefijo] = []
    grupos[prefijo].append(f)

# 3. Procesar cada grupo (MIP)
print(f"Iniciando proceso para canal {canal_nombre}...")
for prefijo, planos_files in grupos.items():
    print(f"  Fusionando muestra {prefijo} ({len(planos_files)} planos)...")
    
    # Cargar los 2 o 3 planos
    planos = [io.imread(os.path.join(ruta_entrada, f)) for f in planos_files]
    
    # Calcular Máxima Intensidad (MIP)
    # np.maximum.reduce funciona aunque los tamaños de matriz sean distintos entre grupos
    mip = np.maximum.reduce(planos)
    
    # Guardar 
    io.imsave(os.path.join(ruta_salida, f"{prefijo}_{canal_nombre}_Final.tif"), 
              mip.astype(np.uint16), check_contrast=False)

print(f"\n ¡PROCESO COMPLETADO! La ruta de salida es: {ruta_salida}")