"""
comparar_umbrales.py
====================
Compara distintos métodos de umbralización para Cavalieri
y muestra visualmente cuánto área se segmenta con cada uno.
Ejecuta esto primero para elegir el mejor umbral antes de
lanzar el Cavalieri definitivo.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, filters, morphology

# ============================================================
# CONFIGURACIÓN
# ============================================================
PIXEL_SIZE        = 1.1364
ESCALA_REDUCCION  = 0.2
PIXEL_SIZE_SCALED = PIXEL_SIZE / ESCALA_REDUCCION
AREA_PIXEL_UM2    = PIXEL_SIZE_SCALED ** 2

RUTA_RAIZ = r"C:\Users\Juan Pedro\Desktop\TFG\ct2\Reconstruccion_v2"

# Canales y cortes representativos para comparar
CANALES_TEST = {
    "C00-DAPI":  [51, 60, 67],   # cortes con media alta
    "C01-IBA-1": [51, 60, 67],
    "C02-GFAP":  [51, 60, 67],
}

# ============================================================
# MÉTODOS DE UMBRALIZACIÓN A COMPARAR
# ============================================================
def aplicar_umbrales(img):
    """Devuelve dict con máscara y área para cada método."""
    resultados = {}

    metodos = {
        "otsu":       filters.threshold_otsu(img),
        "li":         filters.threshold_li(img),
        "triangle":   filters.threshold_triangle(img),
        "yen":        filters.threshold_yen(img),
        "mean":       float(img.mean()),
        "p50":        float(np.percentile(img[img > 0], 50)) if (img > 0).any() else 0,
        "p30":        float(np.percentile(img[img > 0], 30)) if (img > 0).any() else 0,
        "p20":        float(np.percentile(img[img > 0], 20)) if (img > 0).any() else 0,
    }

    for nombre, umbral in metodos.items():
        mascara = img > umbral
        mascara = morphology.remove_small_objects(mascara, min_size=50)
        area_um2 = float(np.sum(mascara)) * AREA_PIXEL_UM2
        pct      = 100 * np.sum(mascara) / img.size
        resultados[nombre] = {
            "umbral":  umbral,
            "area_um2": area_um2,
            "pct":     pct,
            "mascara": mascara,
        }
    return resultados


# ============================================================
# EJECUCIÓN
# ============================================================
for canal, indices_test in CANALES_TEST.items():
    ruta_canal = os.path.join(RUTA_RAIZ, canal)
    archivos   = os.listdir(ruta_canal)

    print(f"\n{'='*60}")
    print(f"Canal: {canal}")
    print(f"{'='*60}")

    fig, axes = plt.subplots(
        len(indices_test), 9,
        figsize=(22, 4 * len(indices_test))
    )
    if len(indices_test) == 1:
        axes = [axes]

    for row, n in enumerate(indices_test):
        match = [f for f in archivos
                 if f.lower().startswith(f"c{n}_") or f.lower().startswith(f"c{n}.")]
        if not match:
            print(f"  Corte {n} no encontrado")
            continue

        img = io.imread(os.path.join(ruta_canal, match[0]))
        if img.ndim == 3:
            img = img[:, :, 0]
        img_r = transform.rescale(
            img, ESCALA_REDUCCION,
            preserve_range=True, anti_aliasing=True
        ).astype(np.uint16)

        resultados = aplicar_umbrales(img_r)

        print(f"\n  Corte c{n}  (min={img_r.min()}, max={img_r.max()}, "
              f"media={img_r.mean():.1f})")
        print(f"  {'Método':<10} {'Umbral':>8} {'Área (µm²)':>14} {'% tejido':>10}")
        print(f"  {'-'*46}")
        for nombre, res in resultados.items():
            print(f"  {nombre:<10} {res['umbral']:>8.1f} "
                  f"{res['area_um2']:>14,.0f} {res['pct']:>9.1f}%")

        # Visualización
        ax = axes[row][0]
        ax.imshow(img_r, cmap="gray")
        ax.set_title(f"c{n} original", fontsize=8)
        ax.axis("off")

        for col, (nombre, res) in enumerate(resultados.items(), start=1):
            ax = axes[row][col]
            ax.imshow(res["mascara"], cmap="Reds")
            ax.set_title(
                f"{nombre}\nu={res['umbral']:.0f} ({res['pct']:.0f}%)",
                fontsize=7)
            ax.axis("off")

    plt.suptitle(f"Comparativa umbrales — {canal}", fontsize=12)
    plt.tight_layout()

    ruta_fig = os.path.join(
        r"C:\Users\Juan Pedro\Desktop\TFG\ct2",
        f"umbrales_{canal.replace('-','_')}.png")
    plt.savefig(ruta_fig, dpi=120, bbox_inches="tight")
    print(f"\n  Figura guardada: {ruta_fig}")
    plt.close()

print("\n✓ Comparativa completada. Revisa las imágenes guardadas.")
print("  Elige el método que mejor delimite el tejido real sin ruido.")
print("  Luego actualiza el parámetro en cavalieri_tfg.py")