"""w en el plano y = 0 de una esfera sana y de la misma esfera sin un casquete,
y su perfil a lo largo del eje z.

    python demos/demo_winding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsd import icosphere, punch_hole, winding_number
from gsd.winding import sample_grid

SALIDA = Path(__file__).resolve().parent.parent / "figuras"

AZUL = "#28598C"
ROJO = "#A6412F"
RADIO_HUECO = 0.6


def corte(ax, mesh, titulo: str, resolucion: int = 220):
    w, extent = sample_grid(mesh, resolution=resolucion, padding=0.2, axis=1, offset=0.0)

    im = ax.imshow(
        w,
        extent=extent,
        origin="lower",
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    ax.contour(w, levels=[0.5], extent=extent, colors="black", linewidths=1.2)

    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("coordenada x")
    ax.set_ylabel("coordenada z")
    ax.set_aspect("equal")
    return im


def perfil(ax, sana, rota):
    t = np.linspace(-1.6, 1.6, 600)
    P = np.stack([np.zeros_like(t), np.zeros_like(t), t], axis=1)

    ax.plot(t, winding_number(sana, P), color=AZUL, lw=1.8, label="esfera cerrada (sana)")
    ax.plot(t, winding_number(rota, P), color=ROJO, lw=1.8, label="con un casquete faltante")

    ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.text(-1.55, 0.54, "w = 0.5 (umbral adentro/afuera)", fontsize=8, color="black", alpha=0.7)
    ax.axvline(1.0, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.text(1.02, -0.28, "polo norte\n(donde está el hueco)", fontsize=8, color="gray")

    ax.set_title("perfil de w a lo largo del eje z", fontsize=11)
    ax.set_xlabel("coordenada z")
    ax.set_ylabel("winding number w")
    ax.set_ylim(-0.4, 1.4)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)


def main() -> None:
    sana = icosphere(4)
    rota = punch_hole(sana, np.array([0.0, 0.0, 1.0]), RADIO_HUECO)

    faltante = 1.0 - rota.total_area / sana.total_area
    centro = winding_number(rota, np.zeros((1, 3)))[0]

    print(f"caras: {sana.n_faces} sana, {rota.n_faces} rota")
    print(f"área faltante: {faltante:.4f}")
    print(f"winding number en el centro de la rota: {centro:.4f}")
    print(f"  (predicho 1 - área faltante = {1.0 - faltante:.4f})")

    fig = plt.figure(figsize=(15, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 0.05, 1.15])
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 1)]
    cax = fig.add_subplot(gs[0, 2])

    im = corte(axes[0], sana, "esfera cerrada")
    corte(axes[1], rota, f"con un casquete faltante (r = {RADIO_HUECO})")
    perfil(fig.add_subplot(gs[0, 3]), sana, rota)

    fig.colorbar(im, cax=cax)
    cax.set_title("w\n(0 a 1)", fontsize=9, pad=8)

    SALIDA.mkdir(exist_ok=True)
    destino = SALIDA / "etapa1_winding.png"
    fig.savefig(destino, dpi=150, bbox_inches="tight")
    print(f"\nfigura escrita en {destino.relative_to(SALIDA.parent)}")


if __name__ == "__main__":
    main()
