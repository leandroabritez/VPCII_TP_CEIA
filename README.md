# Sistema de detección automática de fracturas óseas en radiografías mediante modelos de visión por computadora

## Visión por Computadora II - Trabajo Práctico

Este repositorio contiene la resolución del trabajo práctico de la materia Visión por Computadora II (CEIA - FIUBA).

**Integrantes:**
- Cesar Orellana
- Leandro Britez


## Descripción del problema

La detección de fracturas óseas en estudios radiográficos es una tarea crítica en el ámbito de la salud. En servicios de urgencias o centros con alta demanda, la interpretación puede requerir tiempo y depender de la disponibilidad de especialistas. Un sistema de visión por computadora que asista en la identificación de fracturas puede funcionar como herramienta de apoyo, reduciendo tiempos de evaluación y disminuyendo la posibilidad de pasar por alto lesiones.

El proyecto desarrolla un prototipo capaz de detectar automáticamente la presencia y ubicación de fracturas en radiografías utilizando modelos de Deep Learning preentrenados (Object Detection con bounding boxes).


## Configuración Experimental y Resultados

El proyecto evalúa dos arquitecturas de detección de objetos bajo un diseño experimental estructurado en tres configuraciones, tomando como inspiración metodológica trabajos recientes de literatura sobre la materia:

1. **YOLO11m + RAW (Baseline):** Evaluación directa sobre las imágenes originales para determinar el desempeño nativo de la arquitectura. Alcanzó un **mAP@0.5 de 0.2550** y una inferencia rápida de **10.90 ms**.
2. **YOLO11m + Preprocesamiento:** Aplicación de un pipeline fotométrico (CLAHE + *unsharp masking*). Contrario a antecedentes paralelos, el rendimiento decayó fuertemente a un **mAP@0.5 de 0.0766**.
3. **RT-DETR-L + Preprocesamiento:** Evaluación complementaria utilizando un detector basado en Transformers (*Global-Attention*). Logró un **mAP@0.5 de 0.2475** y un **F1-score de 0.2854**, con una latencia mayor de **41.68 ms**.

**Conclusiones Clave:**
* El preprocesamiento fotométrico intensivo no produce beneficios universales invariantes a la arquitectura subyacente.
* El fuerte desbalance de clases impactó directamente sobre lesiones subrrepresentadas, revelando la fragilidad local.
* RT-DETR evidenció su superioridad para lidiar con el preprocesamiento agresivo de las imágenes y modelar fracturas superpuestas, aunque incurriendo en un mayor coste de rendimiento en milisegundos.

Consulte el paper oficial formato IEEE en la carpeta `LaTeX/` para acceder a la investigación metodológica rigurosa y extendida.


## Dataset

Dataset público: **Bone Fracture Detection** – Roboflow Universe

🔗 https://universe.roboflow.com/veda/bone-fracture-detection-daoon

---

## Estructura del Repositorio

```
VPCII_TP_CEIA/
├── data/
│   └── README.md              # instrucciones de descarga
├── notebooks/
│   ├── 01_EDA.ipynb           # análisis exploratorio
│   ├── 02_train_yolo.ipynb    # fine-tuning YOLOv11 (Local)
│   ├── 02_train_yolo_colab.ipynb # fine-tuning YOLOv11 (Colab Unificado RAW+CLAHE)
│   ├── 03_train_rtdetr_colab.ipynb  # fine-tuning RT-DETR (Colab)
│   └── 04_evaluation.ipynb    # comparación y métricas finales
├── src/
│   ├── dataset.py             # carga de anotaciones y estadísticas
│   ├── evaluate.py            # métricas y curvas PR
│   └── visualize.py           # visualización de detecciones
├── configs/
│   ├── yolov11.yaml           # hiperparámetros YOLOv11
│   └── model2.yaml            # hiperparámetros RT-DETR
├── results/                   # experimentos, plots, checkpoints
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
   - `01_EDA.ipynb` → Explorar el dataset
   - `02_train_yolo.ipynb` → Entrenar YOLOv11
   - `03_train_model2.ipynb` → Entrenar RT-DETR
   - `04_evaluation.ipynb` → Comparar modelos

Los checkpoints y figuras se guardan automáticamente en `results/`.
