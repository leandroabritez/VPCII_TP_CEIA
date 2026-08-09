"""
visualize.py – Visualización de imágenes con anotaciones y detecciones.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


def plot_samples_with_boxes(
    data_dir: Path,
    annotations_df: pd.DataFrame,
    n_samples: int = 6,
    save_path: Path | None = None,
) -> None:
    """
    Muestra una grilla de imágenes del split con sus bounding boxes.

    Args:
        data_dir: Directorio del split (ej. train/).
        annotations_df: DataFrame devuelto por dataset.load_annotations_df().
        n_samples: Número de imágenes a mostrar.
        save_path: Ruta opcional para guardar la figura.
    """
    images_dir = data_dir / "images"
    img_names = annotations_df["image"].unique()[:n_samples]

    cols = min(3, n_samples)
    rows = math.ceil(n_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = np.array(axes).flatten()

    for ax, img_name in zip(axes, img_names):
        img_path = images_dir / f"{img_name}.jpg"
        if not img_path.exists():
            ax.axis("off")
            continue

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ax.imshow(img)

        bboxes = annotations_df[annotations_df["image"] == img_name]
        for _, row in bboxes.iterrows():
            cx, cy, bw, bh = row["cx"], row["cy"], row["bbox_w"], row["bbox_h"]
            x0 = (cx - bw / 2) * w
            y0 = (cy - bh / 2) * h
            rect = patches.Rectangle(
                (x0, y0), bw * w, bh * h,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(x0, y0 - 4, row["class_name"], color="red", fontsize=9, fontweight="bold")

        ax.set_title(img_name, fontsize=9)
        ax.axis("off")

    for ax in axes[len(img_names):]:
        ax.axis("off")

    plt.suptitle("Muestras con anotaciones", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_detections_grid(
    model: Any,
    image_paths: list[str],
    title: str = "Detecciones",
    conf: float = 0.25,
    save_path: Path | None = None,
) -> None:
    """
    Ejecuta inferencia en una lista de imágenes y muestra los resultados en grilla.

    Args:
        model: Modelo ultralytics ya cargado.
        image_paths: Lista de rutas a imágenes.
        title: Título de la figura.
        conf: Umbral de confianza para mostrar detecciones.
        save_path: Ruta opcional para guardar la figura.
    """
    n = len(image_paths)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = np.array(axes).flatten()

    for ax, img_path in zip(axes, image_paths):
        results = model.predict(img_path, conf=conf, verbose=False)
        annotated = results[0].plot()  # BGR numpy array
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        ax.imshow(annotated)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
