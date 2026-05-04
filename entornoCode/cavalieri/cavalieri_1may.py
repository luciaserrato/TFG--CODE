

import os
import csv
import numpy as np
from skimage import io, transform, filters, morphology

# ============================================================
# CONFIGURACIÓN
# ============================================================
PIXEL_SIZE        = 1.1364   # µm por píxel original hallado con Fiji
ESCALA_REDUCCION  = 0.2      # factor de downscale, disminuye procesador y memoria, pero reduce resolución
PIXEL_SIZE_SCALED = PIXEL_SIZE / ESCALA_REDUCCION
AREA_PIXEL_UM2    = PIXEL_SIZE_SCALED ** 2

RUTA_RAIZ  = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\Reconstruccion_v2"
RUTA_SALIDA = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\csv\cavalieri_resultados.csv"

# Canales a analizar y su método de umbralización óptimo
# - DAPI:  Otsu (distribución bimodal clara)
# - IBA-1: Li   (marcaje tenue, Li es más sensible)
# - GFAP:  Li   (marcaje difuso, igual que IBA-1)
CANALES = {
    "C00-DAPI":  "otsu",
    "C01-IBA-1": "li",
    "C02-GFAP":  "li",
}

# Muestreos de Cavalieri
# T = distancia entre cortes en µm (criostato corta cada 20µm, se coge 1 de cada N)
MUESTREOS = {
    "Cavalieri_40um_impares": {
        "indices": [45, 47, 49, 51, 53, 55, 57, 60, 61, 63, 65, 67, 69, 71, 73],
        "T": 40,
        "descripcion": "Cada 2 cortes (impares)"
    },
    "Cavalieri_40um_pares": {
        "indices": [46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 67, 68, 70, 71, 73],
        "T": 40,
        "descripcion": "Cada 2 cortes (pares)"
    },
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
    """Devuelve máscara binaria según el método indicado."""
    if metodo == "otsu":
        umbral = filters.threshold_otsu(img)
    elif metodo == "li":
        umbral = filters.threshold_li(img)
    elif metodo == "triangle":
        umbral = filters.threshold_triangle(img)
    else:
        raise ValueError(f"Método desconocido: {metodo}")
    mascara = img > umbral
    # Eliminar objetos pequeños (ruido < 50 px²)
    mascara = morphology.remove_small_objects(mascara, min_size=50)
    return mascara, float(umbral)


def calcular_cavalieri(canal, metodo_umbral, indices, T):
    """
    Estima el volumen de un canal por el método de Cavalieri.

    V = T × Σ A_i
    donde T es la distancia entre cortes y A_i es el área del corte i.

    Retorna (volumen_um3, lista_areas_um2, n_encontrados)
    """
    ruta_canal = os.path.join(RUTA_RAIZ, canal)
    if not os.path.exists(ruta_canal):
        print(f"  [ERROR] No existe: {ruta_canal}")
        return 0.0, [], 0

    archivos = os.listdir(ruta_canal)
    areas    = []
    umbrales = []
    n_ok     = 0

    for n in indices:
        # Buscar archivo que empiece por c{n} (insensible a mayúsculas)
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

            # Reducir escala
            img_r = transform.rescale(
                img, ESCALA_REDUCCION,
                preserve_range=True, anti_aliasing=True
            ).astype(np.uint16)

            mascara, umbral = umbralizar(img_r, metodo_umbral)
            area_um2 = float(np.sum(mascara)) * AREA_PIXEL_UM2
            areas.append(area_um2)
            umbrales.append(umbral)
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
print(f"Pixel size original: {PIXEL_SIZE} µm")
print(f"Área por píxel:      {AREA_PIXEL_UM2:.4f} µm²")
print("=" * 65)

# Estructura: resultados[canal][muestreo] = (vol, areas, n_ok)
resultados = {canal: {} for canal in CANALES}

for canal, metodo in CANALES.items():
    print(f"\n{'─'*65}")
    print(f"Canal: {canal}  (umbral: {metodo})")
    print(f"{'─'*65}")

    for nombre, cfg in MUESTREOS.items():
        vol, areas, n_ok = calcular_cavalieri(
            canal, metodo, cfg["indices"], cfg["T"])
        resultados[canal][nombre] = (vol, areas, n_ok)

        print(f"\n  [{cfg['descripcion']}]")
        print(f"  Cortes analizados: {n_ok}/{len(cfg['indices'])}")
        print(f"  Áreas por corte (µm²): "
              f"{[f'{a:.0f}' for a in areas if a > 0]}")
        print(f"  Σ áreas = {sum(areas):,.0f} µm²")
        print(f"  Volumen = {vol/1e9:.6f} mm³  ({vol:,.0f} µm³)")

# ── Comparativa entre muestreos por canal ──────────────────────
print(f"\n{'='*65}")
print("COMPARATIVA ENTRE MUESTREOS")
print(f"{'Canal':<15} {'Muestreo':<30} {'Volumen (mm³)':>14} {'Error rel.':>11}")
print(f"{'─'*65}")

filas_csv = [["Canal", "Muestreo", "Descripcion", "T_um",
              "N_cortes", "SumaAreas_um2", "Volumen_mm3", "Error_rel_%"]]

for canal in CANALES:
    vols = {n: resultados[canal][n][0] for n in MUESTREOS}
    # Referencia: promedio de los dos muestreos a 40µm
    v_ref_impares = vols.get("Cavalieri_40um_impares", 0)
    v_ref_pares   = vols.get("Cavalieri_40um_pares",   0)
    v_ref = (v_ref_impares + v_ref_pares) / 2 if (v_ref_impares + v_ref_pares) > 0 else 1

    for nombre, cfg in MUESTREOS.items():
        vol, areas, n_ok = resultados[canal][nombre]
        err = error_relativo(v_ref, vol) if nombre != "Cavalieri_40um_impares" else 0.0
        err_str = f"{err:.2f}%" if not (err != err) else "—"  # nan check

        print(f"  {canal:<13} {cfg['descripcion']:<30} "
              f"{vol/1e9:>13.6f}  {err_str:>10}")

        filas_csv.append([
            canal, nombre, cfg["descripcion"], cfg["T"],
            n_ok, f"{sum(areas):.2f}", f"{vol/1e9:.6f}",
            f"{err:.2f}" if not (err != err) else ""
        ])

    print()

# ── Guardar CSV ────────────────────────────────────────────────
try:
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    with open(RUTA_SALIDA, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(filas_csv)
    print(f"✓ Resultados guardados en: {RUTA_SALIDA}")
except Exception as e:
    print(f"[AVISO] No se pudo guardar CSV: {e}")

print("=" * 65)