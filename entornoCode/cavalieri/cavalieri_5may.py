import os
import csv
import numpy as np
from skimage import io, transform, filters, morphology
from scipy import ndimage

# ============================================================
# CONFIGURACIÓN
# ============================================================
PIXEL_SIZE        = 1.1364
ESCALA_REDUCCION  = 0.2
PIXEL_SIZE_SCALED = PIXEL_SIZE / ESCALA_REDUCCION
AREA_PIXEL_UM2    = PIXEL_SIZE_SCALED ** 2

RUTA_RAIZ  = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\Reconstruccion_v2"
RUTA_SALIDA = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\csv\cavalieri_resultados.csv"

# 🔥 CAMBIO: DAPI usa "tejido"
CANALES = {
    "C00-DAPI":  "tejido",
    "C01-IBA-1": "li",
    "C02-GFAP":  "li",
}

MUESTREOS = {
    "Cavalieri_40um_impares": {
        "indices": [45, 47, 49, 51, 53, 55, 57, 60, 61, 63, 65, 67, 69, 71, 73],
        "T": 40,
        "descripcion": "Cada 2 cortes (impares)"
    }, #solo impares porque hay mas muestras 
    "Cavalieri_60um": {
        "indices": [45, 48, 51, 54, 57, 60, 63, 65, 69, 73],
        "T": 60,
        "descripcion": "Cada 3 cortes (60 µm)"
    },
}

# ============================================================
# FUNCIONES
# ============================================================

def umbralizar(img, metodo):

    # ========================================================
    # 🔥UEVO: MODO TEJIDO PARA DAPI
    # ========================================================
    if metodo == "tejido":
        img = img.astype(np.float32)
        
        # 1. Estimación y resta del ruido de fondo (Background Subtraction)
        # Tomamos el promedio del 10% de los píxeles más oscuros como nivel de ruido
        fondo_estimado = np.mean(img[img < np.percentile(img, 10)]) 
        img = np.maximum(0, img - fondo_estimado) 
        
        # 2. Suavizado Gaussiano para integrar núcleos en una masa continua
        img_s = ndimage.gaussian_filter(img, sigma=3)
        
        # 3. Cálculo de umbral dinámico sobre el 50% superior de la señal
        positivos = img_s[img_s > 0]
        if positivos.size == 0:
            return np.zeros_like(img, dtype=bool), 0.0
            
        umbral = np.percentile(positivos, 50) 
        mascara = img_s > umbral

        # 4. Operaciones morfológicas para consolidar el volumen del tumor
        # Rellena huecos internos y elimina pequeñas motas de ruido
        mascara = morphology.binary_closing(mascara, morphology.disk(10))
        mascara = morphology.remove_small_objects(mascara, min_size=500)

        return mascara, float(umbral)

    # ========================================================
    # MÉTODOS NORMALES
    # ========================================================
    if metodo == "otsu":
        umbral = filters.threshold_otsu(img)

    elif metodo == "li":
        umbral = filters.threshold_li(img)

    elif metodo == "percentil":
        positivos = img[img > 0]
        if positivos.size == 0:
            return np.zeros_like(img, dtype=bool), 0.0
        umbral = np.percentile(positivos, 95)

    else:
        raise ValueError(f"Método desconocido: {metodo}")

    mascara = img > umbral
    mascara = morphology.remove_small_objects(mascara, min_size=50)

    return mascara, float(umbral)


def calcular_cavalieri(canal, metodo_umbral, indices, T):

    ruta_canal = os.path.join(RUTA_RAIZ, canal)
    if not os.path.exists(ruta_canal):
        print(f"  [ERROR] No existe: {ruta_canal}")
        return 0.0, [], 0

    archivos = os.listdir(ruta_canal)
    areas    = []
    n_ok     = 0

    for n in indices:

        match = [f for f in archivos
                 if f.lower().startswith(f"c{n}_") or f.lower().startswith(f"c{n}.")]

        if not match:
            print(f"    [AVISO] Corte {n} no encontrado en {canal}")
            areas.append(0.0)
            continue

        try:
            ruta_img = os.path.join(ruta_canal, match[0])
            img = io.imread(ruta_img)

            if img.ndim == 3:
                img = img[:, :, 0]

            img_r = transform.rescale(
                img, ESCALA_REDUCCION,
                preserve_range=True, anti_aliasing=True
            ).astype(np.uint16)

            mascara, _ = umbralizar(img_r, metodo_umbral)

            area_um2 = float(np.sum(mascara)) * AREA_PIXEL_UM2
            areas.append(area_um2)
            n_ok += 1

        except Exception as e:
            print(f"    [ERROR] Corte {n}: {e}")
            areas.append(0.0)

    volumen_um3 = sum(areas) * T
    return volumen_um3, areas, n_ok


def error_relativo(v_ref, v_comp):
    if v_ref == 0:
        return float("nan")
    return abs(v_ref - v_comp) / v_ref * 100


# ============================================================
# EJECUCIÓN
# ============================================================

print("=" * 65)
print("ESTIMACIÓN DE VOLUMEN — MÉTODO DE CAVALIERI")
print("=" * 65)

resultados = {canal: {} for canal in CANALES}

for canal, metodo in CANALES.items():

    print(f"\n{'─'*65}")
    print(f"Canal: {canal}  (modo: {metodo})")
    print(f"{'─'*65}")

    for nombre, cfg in MUESTREOS.items():

        vol, areas, n_ok = calcular_cavalieri(
            canal, metodo, cfg["indices"], cfg["T"])

        resultados[canal][nombre] = (vol, areas, n_ok)

        print(f"\n  [{cfg['descripcion']}]")
        print(f"  Cortes analizados: {n_ok}/{len(cfg['indices'])}")
        print(f"  Σ áreas = {sum(areas):,.0f} µm²")
        print(f"  Volumen = {vol/1e9:.6f} mm³")

# ============================================================
# CSV
# ============================================================

filas_csv = [["Canal", "Muestreo", "Volumen_mm3"]]

for canal in CANALES:
    for nombre in MUESTREOS:
        vol = resultados[canal][nombre][0]
        filas_csv.append([canal, nombre, f"{vol/1e9:.6f}"])

os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)

with open(RUTA_SALIDA, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(filas_csv)

print(f"\n✓ CSV guardado en: {RUTA_SALIDA}")