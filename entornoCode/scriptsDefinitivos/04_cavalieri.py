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

RUTA_RAIZ   = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\Reconstruccion_v2"
RUTA_SALIDA = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\csv\cavalieri_resultados.csv"

FACTOR_CELULA_NUCLEO = 6.74

CANALES = {
    "C00-DAPI":  "tejido",   # núcleos → se aplica FACTOR_CELULA_NUCLEO
    "C01-IBA-1": "li",
    "C02-GFAP":  "li",
}

MUESTREOS = {
    "Cavalieri_40um_impares": {
        "indices":     [45, 47, 49, 51, 53, 55, 57, 60, 61, 63, 65, 67, 69, 71, 73],
        "T":           40,
        "descripcion": "Cada 2 cortes (impares)"
    },
    "Cavalieri_60um": {
        "indices":     [45, 48, 51, 54, 57, 60, 63, 65, 69, 73],
        "T":           60,
        "descripcion": "Cada 3 cortes (60 µm)"
    },
}

# ============================================================
# FUNCIONES
# ============================================================

def umbralizar(img, metodo):
    """
    Segmenta la imagen según el método indicado.

    'tejido' — modo específico para DAPI:
        Estima el volumen del tejido tumoral completo (núcleos + citoplasma)
        a partir de la señal nuclear. Aplica:
          1. Resta de fondo (percentil 10)
          2. Suavizado gaussiano (sigma=3) para integrar núcleos vecinos
          3. Umbral dinámico (percentil 50 sobre señal positiva)
          4. Cierre morfológico (disk 10) para rellenar citoplasma
          5. Eliminación de objetos pequeños (ruido)

    'li'   — umbral de Li, óptimo para señales fluorescentes tenues
             como IBA-1 y GFAP.
    """
    if metodo == "tejido":
        img = img.astype(np.float32)

        # 1. Resta de fondo
        fondo = np.mean(img[img < np.percentile(img, 10)])
        img   = np.maximum(0, img - fondo)

        # 2. Suavizado para integrar núcleos en masa continua
        img_s = ndimage.gaussian_filter(img, sigma=3)

        # 3. Umbral dinámico
        positivos = img_s[img_s > 0]
        if positivos.size == 0:
            return np.zeros_like(img, dtype=bool), 0.0
        umbral  = np.percentile(positivos, 50)
        mascara = img_s > umbral

        # 4. Morfología para consolidar el volumen tumoral
        mascara = morphology.binary_closing(mascara, morphology.disk(10))
        mascara = morphology.remove_small_objects(mascara, min_size=500)

        return mascara, float(umbral)

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
    """
    Estima el volumen de un canal por el método de Cavalieri:
        V = T × Σ A_i
    """
    ruta_canal = os.path.join(RUTA_RAIZ, canal)
    if not os.path.exists(ruta_canal):
        print(f"  [ERROR] No existe: {ruta_canal}")
        return 0.0, [], 0

    archivos = os.listdir(ruta_canal)
    areas    = []
    n_ok     = 0

    for n in indices:
        match = [f for f in archivos
                 if f.lower().startswith(f"c{n}_") or
                    f.lower().startswith(f"c{n}.")]

        if not match:
            print(f"    [AVISO] Corte {n} no encontrado en {canal}")
            areas.append(0.0)
            continue

        try:
            img = io.imread(os.path.join(ruta_canal, match[0]))
            if img.ndim == 3:
                img = img[:, :, 0]

            img_r = transform.rescale(
                img, ESCALA_REDUCCION,
                preserve_range=True, anti_aliasing=True
            ).astype(np.uint16)

            mascara, _ = umbralizar(img_r, metodo_umbral)

            area_um2 = float(np.sum(mascara)) * AREA_PIXEL_UM2

            # Corrección nuclear → celular para DAPI
            if metodo_umbral == "tejido":
                area_um2 *= FACTOR_CELULA_NUCLEO

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
print(f"Factor corrección nuclear→celular (DAPI): {FACTOR_CELULA_NUCLEO}×")
print("=" * 65)

resultados = {canal: {} for canal in CANALES}

for canal, metodo in CANALES.items():
    print(f"\n{'─'*65}")
    print(f"Canal: {canal}  (modo: {metodo})")
    if metodo == "tejido":
        print(f"  [DAPI] Factor corrección aplicado: ×{FACTOR_CELULA_NUCLEO}")
    print(f"{'─'*65}")

    for nombre, cfg in MUESTREOS.items():
        vol, areas, n_ok = calcular_cavalieri(
            canal, metodo, cfg["indices"], cfg["T"])
        resultados[canal][nombre] = (vol, areas, n_ok)

        print(f"\n  [{cfg['descripcion']}]")
        print(f"  Cortes analizados: {n_ok}/{len(cfg['indices'])}")
        print(f"  Σ áreas = {sum(areas):,.0f} µm²")
        print(f"  Volumen = {vol/1e9:.6f} mm³")

# ── Comparativa entre muestreos ───────────────────────────────
print(f"\n{'='*65}")
print("COMPARATIVA ENTRE MUESTREOS")
print(f"{'Canal':<15} {'Muestreo':<30} {'Volumen (mm³)':>14} {'Error rel.':>11}")
print(f"{'─'*65}")

filas_csv = [["Canal", "Metodo", "Muestreo", "Descripcion",
              "T_um", "N_cortes", "SumaAreas_um2", "Volumen_mm3",
              "Factor_correccion", "Error_rel_%"]]

for canal, metodo in CANALES.items():
    vols    = {n: resultados[canal][n][0] for n in MUESTREOS}
    v_ref   = vols.get("Cavalieri_40um_impares", 1)
    factor  = FACTOR_CELULA_NUCLEO if metodo == "tejido" else 1.0

    for nombre, cfg in MUESTREOS.items():
        vol, areas, n_ok = resultados[canal][nombre]
        err     = error_relativo(v_ref, vol) if nombre != "Cavalieri_40um_impares" else 0.0
        err_str = f"{err:.2f}%" if not (err != err) else "—"
        print(f"  {canal:<13} {cfg['descripcion']:<30} "
              f"{vol/1e9:>13.6f}  {err_str:>10}")
        filas_csv.append([
            canal, metodo, nombre, cfg["descripcion"], cfg["T"],
            n_ok, f"{sum(areas):.2f}", f"{vol/1e9:.6f}",
            f"{factor}", f"{err:.2f}" if not (err != err) else ""
        ])
    print()

# ── Guardar CSV ───────────────────────────────────────────────
os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
with open(RUTA_SALIDA, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(filas_csv)
print(f"✓ CSV guardado en: {RUTA_SALIDA}")