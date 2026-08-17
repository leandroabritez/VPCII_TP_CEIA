# Sistema de detección automática de fracturas óseas en radiografías mediante modelos de visión por computadora

## Visión por Computadora II - Trabajo Práctico

Este repositorio contiene la resolución del trabajo práctico de la materia Visión por Computadora II (CEIA - FIUBA).

**Integrantes:**
- Cesar Orellana
- Leandro Britez


## Descripción del problema

La detección de fracturas óseas en estudios radiográficos es una tarea crítica en el ámbito de la salud. En servicios de urgencias o centros con alta demanda, la interpretación puede requerir tiempo y depender de la disponibilidad de especialistas. Un sistema de visión por computadora que asista en la identificación de fracturas puede funcionar como herramienta de apoyo, reduciendo tiempos de evaluación y disminuyendo la posibilidad de pasar por alto lesiones.

El proyecto desarrolla un prototipo capaz de detectar automáticamente la presencia y ubicación de fracturas en radiografías utilizando modelos de Deep Learning preentrenados (Object Detection con bounding boxes).


## Modelos comparados e Hipótesis Evaluadas

El proyecto compara dos paradigmas fundamentales del estado del arte:
1. **Modelos Convolucionales (YOLOv11):** Presenta ventajas mediante bloques de extracción C3k2 y C2PSA. Procesamiento espacial más apto para texturas óseas pero que requiere *localidad inductiva*.
2. **Transformers de Detección (RT-DETR-L):** Aprovecha la *Atención Global* de los Vision Transformers (ViT) y un *Codificador Híbrido multiescala* para representar la continuidad estructural de todo el hueso, resultando ideal en fallas complejas de rayos X. Elimina NMS basándose en *Selección de Consultas* probabilísticas, reduciendo falsos negativos en lesiones ortopédicas solapadas (ej. fracturas conminutas).

| Modelo | Descripción |
|--------|-------------|
| **YOLO11m** | Modelo \textit{anchor-free} de una sola etapa con priorización espacial C2PSA. |
| **RT-DETR-L** | Detection Transformer. NMS-free, encoder híbrido y auto-atención global clínica. |

La elección final se justifica mediante mAP@0.5, mAP@0.5:0.95, F1, Precision, Recall y latencia de inferencia.


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
   - `01_EDA.ipynb` → Explorar el dataset
   - `02_train_yolo.ipynb` → Entrenar YOLOv11
   - `03_train_model2.ipynb` → Entrenar RT-DETR
   - `04_evaluation.ipynb` → Comparar modelos

Los checkpoints y figuras se guardan automáticamente en `results/`.
