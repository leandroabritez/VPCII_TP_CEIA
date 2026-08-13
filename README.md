# Visión por Computadora II - Trabajo Práctico

Este repositorio contiene la resolución del trabajo práctico de la materia Visión por Computadora II (CEIA - FIUBA).

**Integrantes:**
- Lucia T. Capon Paul
- Cesar Orellana
- Leandro Britez

---

## Título del proyecto
Sistema de detección automática de fracturas óseas en radiografías mediante modelos de visión por computadora

---

## Descripción del problema

La detección de fracturas óseas en estudios radiográficos es una tarea crítica en el ámbito de la salud. En servicios de urgencias o centros con alta demanda, la interpretación puede requerir tiempo y depender de la disponibilidad de especialistas. Un sistema de visión por computadora que asista en la identificación de fracturas puede funcionar como herramienta de apoyo, reduciendo tiempos de evaluación y disminuyendo la posibilidad de pasar por alto lesiones.

El proyecto desarrolla un prototipo capaz de detectar automáticamente la presencia y ubicación de fracturas en radiografías utilizando modelos de Deep Learning preentrenados (Object Detection con bounding boxes).

---

## Modelos comparados

| Modelo | Descripción |
|--------|-------------|
| **YOLOv11** | Modelo principal. Familia YOLO, buen equilibrio precisión/velocidad. |
| **RT-DETR** | Modelo alternativo. Detection Transformer de tiempo real. |

La elección final se justifica mediante mAP@0.5, mAP@0.5:0.95, F1, Precision, Recall y latencia de inferencia.

---

## Dataset

Dataset público: **Bone Fracture Detection** – Roboflow Universe

🔗 https://universe.roboflow.com/veda/bone-fracture-detection-daoon

> El mismo dataset está disponible en Kaggle ("Bone Fracture Detection Computer Vision Project") con idénticas imágenes y anotaciones.  
> Ver instrucciones de descarga en [`data/README.md`](data/README.md).

---

## Estructura del Repositorio

```
VPCII_TP_CEIA/
├── data/
│   └── README.md              # instrucciones de descarga
├── notebooks/
│   ├── 00_preprocessing.ipynb # preprocesamiento de radiografías
│   ├── 01_EDA.ipynb           # análisis exploratorio
│   ├── 02_train_yolo.ipynb    # fine-tuning YOLOv11
│   ├── 03_train_model2.ipynb  # fine-tuning RT-DETR
│   └── 04_evaluation.ipynb    # comparación y métricas finales
├── src/
│   ├── dataset.py             # carga de anotaciones y estadísticas
│   ├── preprocessing.py       # funciones reutilizables de preprocesamiento
│   ├── evaluate.py            # métricas y curvas PR
│   └── visualize.py           # visualización de detecciones
├── configs/
│   ├── yolov11.yaml           # hiperparámetros YOLOv11
│   └── model2.yaml            # hiperparámetros RT-DETR
├── results/                   # experimentos, plots, checkpoints
├── Papers/                    # bibliografía PDF
├── LaTeX/                     # paper final formato IEEE
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
```

---

## Instalación

### Con `uv` (recomendado)
```bash
pip install uv
uv sync
```

### Con pip
```bash
pip install -r requirements.txt
```

---

## Uso

1. Descargar el dataset → instrucciones en `data/README.md`
2. Ejecutar los notebooks en orden:
   - `00_preprocessing.ipynb` → Crear el dataset preprocesado (opcional, para el experimento con CLAHE y unsharp masking)
   - `01_EDA.ipynb` → Explorar el dataset
   - `02_train_yolo.ipynb` → Entrenar YOLOv11
   - `03_train_model2.ipynb` → Entrenar RT-DETR
   - `04_evaluation.ipynb` → Comparar modelos

Los checkpoints y figuras se guardan automáticamente en `results/`.
