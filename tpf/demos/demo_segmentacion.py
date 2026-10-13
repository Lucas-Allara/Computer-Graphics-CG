"""Corte de grafo contra umbral simple sobre un w sintético: una transición
suave en x más ruido independiente en cada tetraedro. Se muestra una rebanada
coloreada por acierto o error.

    python demos/demo_segmentacion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsd.segmentation import (
    data_term,
    estimate_sigma,
    min_cut_labels,
    smoothness_weights,
    tet_adjacency,
)
from gsd.tetmesh import tet_grid

SALIDA = Path(__file__).resolve().parent.parent / "figuras"

VERDE = "#2E7D46"
ROJO = "#A6412F"
GRIS = "#8A8A8A"


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)

    mesh = tet_grid(n=10, size=2.0)
    tet_a, tet_b, area = tet_adjacency(mesh)
    centros = mesh.tet_centroids

    w_real = 0.5 * (1 - np.tanh(centros[:, 0] / 0.35))
    rng = np.random.default_rng(3)
    w_ruidoso = np.clip(w_real + rng.normal(0, 0.22, size=len(w_real)), -0.4, 1.4)
    verdad = centros[:, 0] < 0

    u_adentro, u_afuera = data_term(w_ruidoso)
    sigma = estimate_sigma(w_ruidoso, tet_a, tet_b)
    print(f"sigma estimado (percentil 85 de |Δw| entre vecinos): {sigma:.4f}")

    gammas = (0.0, 5.0, 40.0)
    etiquetas, precisiones = [], []
    for gamma in gammas:
        v = gamma * smoothness_weights(w_ruidoso, tet_a, tet_b, area, sigma)
        et = min_cut_labels(u_adentro, u_afuera, tet_a, tet_b, v)
        etiquetas.append(et)
        precisiones.append(float(np.mean(et == verdad)))
        print(f"  gamma={gamma:6.1f}  accuracy={precisiones[-1]:.4f}  "
              f"errores={(et != verdad).sum()}/{len(et)}")

    rebanada = np.abs(centros[:, 2]) < 0.11
    x, y = centros[rebanada, 0], centros[rebanada, 1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    sc = axes[0].scatter(x, y, c=w_ruidoso[rebanada], cmap="RdBu_r", vmin=0, vmax=1, s=14)
    axes[0].set_title("$w$ con ruido incoherente\n(entrada)")
    fig.colorbar(sc, ax=axes[0], fraction=0.046)

    for ax, gamma, et, acc in zip(axes[1:], gammas, etiquetas, precisiones):
        correcto = et[rebanada] == verdad[rebanada]
        ax.scatter(x[correcto], y[correcto], c=VERDE, s=14, label="correcto")
        ax.scatter(x[~correcto], y[~correcto], c=ROJO, s=22, marker="x",
                   label="error")
        etiqueta = "umbral (γ=0)" if gamma == 0 else f"corte de grafo (γ={gamma:g})"
        ax.set_title(f"{etiqueta}\naccuracy {acc:.1%}")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Segmentación por corte de grafo contra umbral simple, sobre "
                 "ruido incoherente entre vecinos", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(SALIDA / "segmentacion_graphcut.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigura en {SALIDA.relative_to(SALIDA.parent)}/segmentacion_graphcut.png")


if __name__ == "__main__":
    main()
