"""Funciones reutilizables para el preprocesamiento de radiografías."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parámetros del preprocesamiento determinista de una radiografía."""

    low_percentile: float = 1.0
    high_percentile: float = 99.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    unsharp_sigma: float = 1.2
    unsharp_amount: float = 0.4
    jpeg_quality: int = 95


def preprocess_xray(
    image_bgr: np.ndarray, config: PreprocessingConfig = PreprocessingConfig()
) -> np.ndarray:
    """Normaliza, mejora el contraste y afila suavemente una radiografía.

    La salida conserva tres canales iguales para ser compatible con los modelos
    preentrenados que reciben imágenes RGB.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("La imagen de entrada está vacía o no pudo leerse.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(gray, (config.low_percentile, config.high_percentile))

    if high <= low:
        normalized = gray.copy()
    else:
        clipped = np.clip(gray, low, high)
        normalized = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )
    enhanced = clahe.apply(normalized)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), config.unsharp_sigma)
    sharpened = cv2.addWeighted(
        enhanced,
        1 + config.unsharp_amount,
        blurred,
        -config.unsharp_amount,
        0,
    )
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def create_preprocessed_dataset(
    data_root: Path,
    output_root: Path,
    config: PreprocessingConfig = PreprocessingConfig(),
    splits: tuple[str, ...] = ("train", "valid", "test"),
) -> list[dict[str, int | str]]:
    """Crea un dataset preprocesado sin cambiar los nombres ni las etiquetas.

    La función rechaza una carpeta de salida existente para no sobrescribir
    experimentos anteriores. El preprocesamiento no cambia la geometría, por lo
    que las cajas YOLO se copian sin modificación.
    """
    if not data_root.is_dir():
        raise FileNotFoundError(f"No se encontró el dataset en: {data_root}")
    if output_root.exists():
        raise FileExistsError(
            f"La carpeta de salida ya existe: {output_root}. "
            "Elegir otra carpeta o revisar el experimento anterior."
        )

    for split in splits:
        (output_root / split / "images").mkdir(parents=True, exist_ok=False)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=False)

    with (data_root / "data.yaml").open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)
    
    # ---------------------------------------------------------
    # PARCHE: Eliminar la clase 'humerus' de data.yaml (unificación)
    # ---------------------------------------------------------
    old_names = data_config.get("names", [])
    if 'humerus' in old_names:
        new_names = [n for n in old_names if n != 'humerus']
        data_config['names'] = new_names
        data_config['nc'] = len(new_names)
    
    with (output_root / "data.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(data_config, file, sort_keys=False, allow_unicode=True)

    summary: list[dict[str, int | str]] = []
    for split in splits:
        input_images_dir = data_root / split / "images"
        input_labels_dir = data_root / split / "labels"
        output_images_dir = output_root / split / "images"
        output_labels_dir = output_root / split / "labels"
        image_paths = sorted(input_images_dir.glob("*.jpg"))

        for image_path in tqdm(image_paths, desc=f"Procesando {split}"):
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            processed_bgr = preprocess_xray(image_bgr, config)
            output_image_path = output_images_dir / image_path.name
            saved = cv2.imwrite(
                str(output_image_path),
                processed_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
            )
            if not saved:
                raise OSError(f"No se pudo guardar la imagen: {output_image_path}")

        # Mapa de unificación para eliminar humerus aislada y juntarla con humerus fracture (y bajar el resto un id)
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 5}

        for label_path in sorted(input_labels_dir.glob("*.txt")):
            # shutil.copy2(label_path, output_labels_dir / label_path.name)
            # ---------------------------------------------------------
            # PARCHE: Aplicamos el re-mapeo dinámico mientras escribimos
            # ---------------------------------------------------------
            content = label_path.read_text(encoding='utf-8').strip()
            if not content:
                # Archivo vacío (imagen negativa)
                shutil.copy2(label_path, output_labels_dir / label_path.name)
                continue
            
            new_lines = []
            for line in content.split('\n'):
                parts = line.strip().split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                new_cls = mapping.get(old_cls, old_cls)
                new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
            
            (output_labels_dir / label_path.name).write_text('\n'.join(new_lines) + '\n', encoding='utf-8')

        summary.append(
            {
                "split": split,
                "imagenes_procesadas": len(image_paths),
                "archivos_etiquetas": len(list(output_labels_dir.glob("*.txt"))),
            }
        )

    return summary
