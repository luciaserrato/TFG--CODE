Reconstrucción 3D y Estereología de Glioblastoma en Python

Pipeline completo para procesamiento de imágenes de inmunofluorescencia, reconstrucción volumétrica 3D y estimación estereológica mediante el método de Cavalieri aplicado a modelos murinos de glioblastoma.

Desarrollado como parte del Trabajo Fin de Grado en Ingeniería Biomédica.

Características principales
Fusión automática de Z-stacks mediante Maximum Intensity Projection (MIP)
Normalización y alineamiento automático de cortes histológicos
Interpolación volumétrica 3D en eje Z
Reconstrucción compatible con Napari
Segmentación GFAP configurable
Estimación volumétrica mediante método de Cavalieri
Exportación de volúmenes .npz
Exportación de resultados cuantitativos .csv
Estructura del proyecto
├── 01_merge.py
├── 02_normalization.py
├── 03_cavalieri.py
└── README.md

Requisitos

Python 3.10 o superior.

Instalar dependencias:

pip install numpy scipy scikit-image opencv-python napari tifffile

Flujo completo del pipeline
1. Fusionar Z-stacks

Script: 01_merge.py
Función: Fusiona múltiples planos confocales en una sola imagen 2D mediante Maximum Intensity Projection (MIP).

Entrada esperada: Carpeta con TIFFs organizados por canal:
C00-DAPI/
C01-IBA-1/
C02-GFAP/

Salida: Imágenes MIP finales:
c45_DAPI_Final.tif
c45_IBA-1_Final.tif
c45_GFAP_Final.tif

2. Normalización y reconstrucción 3D

Script: 02_normalization.py
Función: Reduce resolución para optimizar memoria, Alinea cortes histológicos, Interpola eje Z, 
Genera volumen 3D, Guarda metadatos físicos, Exporta volumen .npz

Parámetros importantes
CARPETA_ENTRADA, ARCHIVO_SALIDA, PIXEL_SIZE_UM (Obtener mediante Fiji), ESCALA_REDUCCION,
MICRAS_ENTRE_CORTES, INDICES_VALIDOS

Salida:volumenes.npz

3. Estimación volumétrica (Cavalieri)

Script: 03_cavalieri.py
Función: Calcula volumen tumoral 

Organización de datos recomendada
Reconstruccion_v2/
│
├── C00-DAPI/
├── C01-IBA-1/
└── C02-GFAP/

Resultados obtenidos

El pipeline fue validado sobre modelos murinos GL261:

Reconstrucción 3D volumétrica
Segmentación multicanal
Estimación estereológica
Compatibilidad con Napari
Limitaciones
Volúmenes grandes requieren mucha RAM
Napari puede fallar en GPUs integradas
GFAP difuso requiere ajuste manual de umbral
Pérdida de cortes histológicos afecta precisión de Cavalieri
Autor

Lucía Serrato
Trabajo Fin de Grado — Ingeniería Biomédica
Universidad de Sevilla

Licencia: Uso académico y de investigación.

