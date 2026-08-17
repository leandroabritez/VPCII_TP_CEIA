"""
evaluate.py – Métricas y comparación de modelos de detección de objetos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_metrics_summary(models: dict[str, Any]) -> pd.DataFrame:
    """
    Construye un DataFrame comparativo de métricas a partir de los resultados
    de validación de ultralytics.

    Args:
        models: Dict {model_name: ultralytics validation results object}

    Returns:
        pd.DataFrame con columnas: mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1
    """
    rows = {}
    for name, result in models.items():
        b = result.box
        precision = float(b.mp)
        recall    = float(b.mr)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        rows[name] = {
            "mAP@0.5":      round(float(b.map50), 4),
            "mAP@0.5:0.95": round(float(b.map),   4),
            "Precision":    round(precision,        4),
            "Recall":       round(recall,           4),
            "F1":           round(f1,               4),
        }
    return pd.DataFrame(rows).T


def plot_pr_curves(
    results_dict: dict[str, Any],
    save_path: Path | None = None,
) -> None:
    """
    Dibuja las curvas Precision-Recall para cada modelo.

    Args:
        results_dict: Dict {model_name: ultralytics validation results object}
        save_path: Ruta opcional para guardar la figura.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, result in results_dict.items():
        # ultralytics almacena curvas en result.box.curves_results
        # Intentamos acceder a precisión y recall por umbral si está disponible
        try:
            px = result.box.curves_results[0]   # recall values
            py = result.box.curves_results[1]   # precision values
            ax.plot(px, py, label=name, linewidth=2)
        except Exception:
            # fallback: graficar punto único
            b = result.box
            ax.scatter([float(b.mr)], [float(b.mp)], label=name, s=120, zorder=5)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Curvas Precision-Recall – Test Set", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
