"""w a lo largo de un eje que pasa por dos esferas, con las dos completas, con
un agujero en la segunda y con la segunda borrada.

Un agujero degrada w cerca de la pieza; borrarla la saca de la suma, y su
aporte desaparece en todos lados.

    python demos/demo_pieza_faltante.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsd import icosphere, merge_meshes, punch_hole, winding_number
from gsd.mesh import Mesh

SALIDA = Path(__file__).resolve().parent.parent / "figuras"

NEGRO = "#14161d"
ROJO = "#A6412F"
AZUL = "#2F6FB5"


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)

    a = icosphere(2)
    b = icosphere(2)
    b = Mesh(b.V + np.array([4.0, 0.0, 0.0]), b.F)

    xs = np.linspace(-1.3, 5.3, 300)
    pts = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])

    completas = merge_meshes([a, b])
    w_completas = winding_number(completas, pts)

    b_agujero = punch_hole(b, center=np.array([4.0, 0.0, 1.0]), radius=1.3)
    con_agujero = merge_meshes([a, b_agujero])
    w_agujero = winding_number(con_agujero, pts)

    w_borrada = winding_number(a, pts)

    lejos = xs < 1.5
    diff_lejos = np.abs(w_completas[lejos] - w_borrada[lejos]).max()
    i4 = np.argmin(np.abs(xs - 4.0))
    print(f"w en x=4: completas={w_completas[i4]:.4f}  "
          f"con agujero={w_agujero[i4]:.4f}  borrada={w_borrada[i4]:.6f}")
    print(f"diferencia máxima cerca de A (x<1.5) al borrar B: {diff_lejos:.2e}")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(xs, w_completas, color=NEGRO, lw=2.0, label="las dos piezas completas")
    ax.plot(xs, w_agujero, color=AZUL, lw=2.0, ls="--",
            label="B con un agujero (sigue estando)")
    ax.plot(xs, w_borrada, color=ROJO, lw=2.2,
            label="B borrada por completo")

    ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax.axhline(1, color="black", lw=0.6, alpha=0.4)
    ax.axvline(0, color="black", lw=0.6, alpha=0.25)
    ax.axvline(4, color="black", lw=0.6, alpha=0.25)
    ax.annotate("centro de A", (0, 1.06), ha="center", fontsize=9, color="black")
    ax.annotate("centro de B", (4, 1.06), ha="center", fontsize=9, color="black")

    ax.set_xlabel("posición a lo largo del eje")
    ax.set_ylabel("$w(x)$")
    ax.set_title("El aporte de una pieza a $w$ en otro lado no depende de esa "
                 "pieza\n(borrarla la saca de la suma; un agujero la deja, "
                 "degradada, en la suma)", fontsize=11)
    ax.legend(loc="center left", fontsize=9, framealpha=0.95)
    fig.tight_layout()

    destino = SALIDA / "pieza_faltante.png"
    fig.savefig(destino, dpi=145)
    plt.close(fig)
    print(f"\nfigura en {destino.relative_to(SALIDA.parent)}")


if __name__ == "__main__":
    main()
