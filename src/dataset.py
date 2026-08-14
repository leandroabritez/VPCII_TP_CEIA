"""
dataset.py – Utilidades para carga y preparación del dataset de fracturas óseas.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def load_annotations_df(split_dir: Path, data_root: Path) -> pd.DataFrame:
    """
    Lee todas las anotaciones YOLO-format (.txt) de un split dado y devuelve
    un DataFrame con una fila por anotación.

    Args:
        split_dir: Directorio del split (p.ej. data_root / 'train').
        data_root: Directorio raíz del dataset que contiene data.yaml.

    Returns:
        pd.DataFrame con una fila por anotación.
    """
    import yaml

    labels_dir = split_dir / "labels"

    # Leer el data.yaml directamente desde el root del dataset
    yaml_path = data_root / "data.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    names = data_config["names"]
    # Soporte para names como lista o como dict {id: nombre}
    if isinstance(names, dict):
        class_map = {int(k): v for k, v in names.items()}
    else:
        class_map = {i: n for i, n in enumerate(names)}

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


def audit_annotations(annotations_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    """Resume errores geométricos y de clase en anotaciones normalizadas YOLO.

    Una caja YOLO válida debe tener clase en ``[0, num_classes)``, centro dentro
    de la imagen, tamaño positivo y bordes dentro del intervalo normalizado.
    """
    if annotations_df.empty:
        return pd.DataFrame(
            [{"annotations": 0, "invalid_class": 0, "invalid_center": 0,
              "invalid_size": 0, "out_of_bounds": 0, "invalid_any": 0}]
        )

    df = annotations_df.copy()
    invalid_class = ~df["class_id"].between(0, num_classes - 1)
    invalid_center = ~df["cx"].between(0, 1) | ~df["cy"].between(0, 1)
    invalid_size = (df["bbox_w"] <= 0) | (df["bbox_h"] <= 0)
    out_of_bounds = (
        (df["cx"] - df["bbox_w"] / 2 < 0)
        | (df["cy"] - df["bbox_h"] / 2 < 0)
        | (df["cx"] + df["bbox_w"] / 2 > 1)
        | (df["cy"] + df["bbox_h"] / 2 > 1)
    )
    invalid_any = invalid_class | invalid_center | invalid_size | out_of_bounds

    return pd.DataFrame(
        [{
            "annotations": len(df),
            "invalid_class": int(invalid_class.sum()),
            "invalid_center": int(invalid_center.sum()),
            "invalid_size": int(invalid_size.sum()),
            "out_of_bounds": int(out_of_bounds.sum()),
            "invalid_any": int(invalid_any.sum()),
        }]
    )


def audit_images(images_dir: Path) -> pd.DataFrame:
    """Informa cuántas imágenes JPG pueden abrirse con OpenCV."""
    image_paths = sorted(images_dir.glob("*.jpg"))
    unreadable = [path.name for path in image_paths if cv2.imread(str(path)) is None]
    return pd.DataFrame(
        [{
            "images": len(image_paths),
            "unreadable": len(unreadable),
            "unreadable_files": ", ".join(unreadable) if unreadable else "-",
        }]
    )


def find_cross_split_duplicates(
    data_root: Path, splits: Iterable[str] = ("train", "valid", "test")
) -> pd.DataFrame:
    """Detecta imágenes idénticas entre particiones mediante SHA-256.

    El análisis identifica duplicados binarios exactos. No detecta versiones
    redimensionadas, recortadas o comprimidas de una misma radiografía.
    """
    files_by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for split in splits:
        for image_path in sorted((data_root / split / "images").glob("*.jpg")):
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            files_by_hash[digest].append((split, image_path.name))

    records = []
    for digest, files in files_by_hash.items():
        present_splits = sorted({split for split, _ in files})
        if len(present_splits) > 1:
            records.append(
                {
                    "sha256": digest,
                    "splits": ", ".join(present_splits),
                    "files": "; ".join(f"{split}/{name}" for split, name in files),
                }
            )

    return pd.DataFrame(records, columns=["sha256", "splits", "files"])
