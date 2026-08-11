"""
dataset.py – Utilidades para carga y preparación del dataset de fracturas óseas.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def load_annotations_df(split_dir: Path, data_config: dict) -> pd.DataFrame:
    """
    Lee todas las anotaciones YOLO-format (.txt) de un split dado y devuelve
    un DataFrame con una fila por anotación.

    Args:
        split_dir: Directorio del split.
        data_config: Configuración del dataset obtenida desde data.yaml.

    Returns:
        pd.DataFrame con una fila por anotación.
    """

    labels_dir = split_dir / "labels"

    # Mapeo de class_id a nombre utilizando data.yaml
    class_map = {
        class_id: class_name
        for class_id, class_name in enumerate(data_config["names"])
    }

    records = []

    for label_path in sorted(labels_dir.glob("*.txt")):

        img_name = label_path.stem

        with open(label_path) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                class_id = int(parts[0])

                cx, cy, w, h = map(
                    float,
                    parts[1:5]
                )

                records.append({
                    "image": img_name,
                    "class_id": class_id,
                    "class_name": class_map.get(
                        class_id,
                        str(class_id)
                    ),
                    "cx": cx,
                    "cy": cy,
                    "bbox_w": w,
                    "bbox_h": h,
                })

    return pd.DataFrame(records)


def get_image_stats(images_dir: Path) -> pd.DataFrame:
    """
    Calcula estadísticas por imagen: resolución, relación de aspecto, brillo medio.

    Args:
        images_dir: Directorio con las imágenes.

    Returns:
        pd.DataFrame con una fila por imagen.
    """
    records = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        records.append({
            "image": img_path.stem,
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 3),
            "mean_brightness": round(float(img.mean()), 2),
            "std_brightness":  round(float(img.std()), 2),
        })
    return pd.DataFrame(records)
