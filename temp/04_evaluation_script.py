import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
import time
import torch
from ultralytics import YOLO, RTDETR

from src.evaluate import compute_metrics_summary, plot_pr_curves
from src.visualize import plot_detections_grid

plt.style.use('seaborn-v0_8-whitegrid')
RESULTS_DIR = Path('../results')
DATA_YAML   = Path('../data/bone-fracture-detection-daoon-1/data.yaml')
print('Setup ok.')

yolo_ckpt   = RESULTS_DIR / 'exp1_yolo11_raw_colab' / 'weights' / 'best.pt'
rtdetr_ckpt = RESULTS_DIR / 'resultados_rtdetr/rtdetr_fracture'  / 'weights' / 'best.pt'

assert yolo_ckpt.exists(),   f'Checkpoint YOLO no encontrado: {yolo_ckpt}'
assert rtdetr_ckpt.exists(), f'Checkpoint RT-DETR no encontrado: {rtdetr_ckpt}'

yolo_model   = YOLO(str(yolo_ckpt))
rtdetr_model = RTDETR(str(rtdetr_ckpt))
print('Modelos cargados.')

yolo_metrics   = yolo_model.val(data=str(DATA_YAML), split='test')
rtdetr_metrics = rtdetr_model.val(data=str(DATA_YAML), split='test')

summary = compute_metrics_summary(
    models   = {'YOLOv11': yolo_metrics, 'RT-DETR': rtdetr_metrics}
)
print(summary)

fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('off')
table = ax.table(
    cellText=summary.values,
    colLabels=summary.columns,
    rowLabels=summary.index,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)
plt.title('Comparación de métricas – Test Set', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '04_metrics_table.png', dpi=150, bbox_inches='tight')
plt.show()

plot_pr_curves(
    results_dict={'YOLOv11': yolo_metrics, 'RT-DETR': rtdetr_metrics},
    save_path=RESULTS_DIR / '04_pr_curves.png'
)

import glob, random

test_images = glob.glob('../data/bone-fracture-detection-daoon-1/test/images/*.jpg')
sample_img  = random.choice(test_images)
N = 20  # repeticiones para promediar

def measure_latency(model, img_path, n=N):
    # Warm up
    model.predict(img_path, verbose=False)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(img_path, verbose=False)
    return (time.perf_counter() - t0) / n * 1000  # ms

yolo_ms   = measure_latency(yolo_model, sample_img)
rtdetr_ms = measure_latency(rtdetr_model, sample_img)

print(f'YOLOv11  – latencia promedio: {yolo_ms:.1f} ms')
print(f'RT-DETR  – latencia promedio: {rtdetr_ms:.1f} ms')

plot_detections_grid(
    model=yolo_model,
    image_paths=test_images[:6],
    title='YOLOv11 – Detecciones en Test Set',
    save_path=RESULTS_DIR / '04_yolo_detections.png'
)

plot_detections_grid(
    model=rtdetr_model,
    image_paths=test_images[:6],
    title='RT-DETR – Detecciones en Test Set',
    save_path=RESULTS_DIR / '04_rtdetr_detections.png'
)

