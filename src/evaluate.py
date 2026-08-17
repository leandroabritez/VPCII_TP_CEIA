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
    Dibuja las curvas Precision-Recall continuas para cada modelo.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, result in results_dict.items():
        # Extraer el Item 0 (Recall vs Precision)
        pr_item = result.box.curves_results[0]
        
        recall = pr_item[0]               # shape (1000,)
        precision_matrix = pr_item[1]     # shape (6, 1000)
        
        # Promedio sobre todas las clases
        precision_mean = precision_matrix.mean(axis=0)
        
        map50 = float(result.box.map50)
        ax.plot(
            recall,
            precision_mean,
            label=f"{name} (mAP@0.5 = {map50:.3f})",
            linewidth=2.5,
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Curvas Precision-Recall – Test Set", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower left")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Curvas guardadas exitosamente en: {save_path}")

    plt.show()
