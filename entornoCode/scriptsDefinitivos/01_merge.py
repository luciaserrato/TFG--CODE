import os
import numpy as np
from skimage import io

# 
#canal_nombre = "DAPI"  # C00- DAPI, C01-Iba-1, C02-GFAP
canal_nombre = "IBA-1"  # C00- DAPI, C01-Iba-1, C02-GFAP
#RUTA DE ENTRADA DONDE ESTÁN LOS PLANOS DE CADA CANAL
ruta_entrada = rf"" 
#RUTA DE SALIDA DONDE SE GUARDARÁN LOS MIP FINALES DE CADA CANAL
ruta_salida = rf"\RESULTADOS_MIP_{canal_nombre}"

if not os.path.exists(ruta_salida):
    os.makedirs(ruta_salida)

# 1. Leer archivos
archivos = sorted([f for f in os.listdir(ruta_entrada) if f.endswith('.tif')])

# 2. Agrupar por muestra 
grupos = {}
for f in archivos:
    prefijo = f.split('--')[0] 
    if prefijo not in grupos:
        grupos[prefijo] = []
    grupos[prefijo].append(f)

# 3. Procesar cada grupo (MIP)
print(f"Iniciando proceso para canal {canal_nombre}...")
for prefijo, planos_files in grupos.items():
    print(f"  Fusionando muestra {prefijo} ({len(planos_files)} planos)...")
    
    # Cargar los planos
    planos = [io.imread(os.path.join(ruta_entrada, f)) for f in planos_files]
    
    # Calcular Máxima Intensidad (MIP)
    mip = np.maximum.reduce(planos)
    
    # Guardar 
    io.imsave(os.path.join(ruta_salida, f"{prefijo}_{canal_nombre}_Final.tif"), 
              mip.astype(np.uint16), check_contrast=False)

print(f"\n ¡PROCESO COMPLETADO! La ruta de salida es: {ruta_salida}")