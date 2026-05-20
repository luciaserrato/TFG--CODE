# Reconstrucción 3D y Estereología de Glioblastoma en Python

Pipeline completo para procesamiento de imágenes de inmunofluorescencia, reconstrucción volumétrica 3D y estimación estereológica mediante el método de Cavalieri aplicado a modelos murinos de glioblastoma.

Desarrollado como parte del Trabajo Fin de Grado en Ingeniería Biomédica.

---

# Características principales

- Fusión automática de Z-stacks mediante **Maximum Intensity Projection (MIP)**
- Normalización y alineamiento automático de cortes histológicos
- Interpolación volumétrica 3D en eje Z
- Reconstrucción compatible con Napari
- Segmentación GFAP configurable
- Estimación volumétrica mediante método de Cavalieri
- Exportación de volúmenes `.npz`
- Exportación de resultados cuantitativos `.csv`

---

# Estructura del proyecto

```bash
├── 01_merge.py
├── 02_normalization.py
├── 03_visualization.py
├── 04_cavalieri.py
├── requirements.txt
└── README.md
