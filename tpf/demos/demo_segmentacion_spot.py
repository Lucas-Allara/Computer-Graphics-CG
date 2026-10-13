"""Spot con cinco agujeros: entrada, umbral (γ=0), corte de grafo, corte de
grafo con las tapas suavizadas y el original. Abajo, la cara de cerca.

A la entrada se le sacan los vértices sueltos que deja ``random_holes``: son
puntos de la vaca real y le darían al método información que no tiene.

    python demos/demo_segmentacion_spot.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsd import Mesh, connected_components, load_obj
from gsd.conforming import conforming_tet_mesh
from gsd.fairing import fair_filled_patches
from gsd.segmentation import graphcut_segment
from gsd.shapes import random_holes

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "figuras"
MODELO = RAIZ / "assets" / "spot" / "spot_triangulated.obj"

ROJO = np.array([0.65, 0.26, 0.18])
AZUL = np.array([0.16, 0.35, 0.55])
VERDE = np.array([0.20, 0.47, 0.27])
VIOLETA = np.array([0.42, 0.30, 0.58])
GRIS = np.array([0.55, 0.58, 0.60])

TAMANO = 1.35


def cargar_spot() -> Mesh:
    m = load_obj(MODELO)
    lo, hi = m.bounding_box()
    escala = TAMANO / float((hi - lo).max())
    V = (m.V - (lo + hi) / 2) * escala
    # El modelo trae el eje vertical en y; se permuta para que quede parada.
    return Mesh(np.column_stack([V[:, 0], -V[:, 2], V[:, 1]]), m.F)


def dibujar(ax, V, F, color, titulo: str, limites):
    if len(F) == 0:
        ax.set_title(f"{titulo}\n(vacío)", fontsize=10)
        ax.set_axis_off()
        return

    tris = V[F]
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    luz = np.array([0.5, 0.35, 0.79])
    luz /= np.linalg.norm(luz)
    intensidad = 0.30 + 0.70 * np.clip(n @ luz, 0.0, 1.0)

    ax.add_collection3d(Poly3DCollection(
        tris, facecolors=np.clip(color[None, :] * intensidad[:, None], 0, 1),
        edgecolors="none",
    ))

    lo, hi = limites
    centro = 0.5 * (lo + hi)
    radio = 0.62 * float((hi - lo).max())
    for k, eje in enumerate("xyz"):
        getattr(ax, f"set_{eje}lim")(centro[k] - radio, centro[k] + radio)

    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.view_init(elev=14, azim=-58)
    ax.set_proj_type("persp", focal_length=0.7)
    ax.set_axis_off()
    ax.set_title(titulo, fontsize=10)


def cara_de_cerca(V, F, color) -> np.ndarray | None:
    """La cabeza de cerca, renderizada con pyvista (de cerca, matplotlib
    ordena mal las caras). Sin pyvista devuelve None."""
    try:
        import pyvista as pv
    except ImportError:
        return None
    pv.OFF_SCREEN = True
    malla = pv.PolyData(V, np.hstack([np.full((len(F), 1), 3), F]).ravel())
    p = pv.Plotter(off_screen=True, window_size=(700, 640))
    p.set_background("white")
    p.add_mesh(malla, color=color, specular=0.25)
    # La cabeza mira hacia +y; se la ve de tres cuartos, del lado del hocico.
    p.camera_position = [(1.2, 0.9, 0.45), (0.0, 0.3, 0.3), (0.0, 0.0, 1.0)]
    imagen = p.screenshot(return_img=True)
    p.close()
    return imagen


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    spot = cargar_spot()
    limites = spot.bounding_box()

    rota = random_holes(spot, target_fraction=0.22, n_holes=5, seed=11)
    usados, F_compacta = np.unique(rota.F, return_inverse=True)
    rota = Mesh(rota.V[usados], F_compacta.reshape(-1, 3))
    faltante = 1.0 - rota.total_area / spot.total_area
    print(f"Spot: {spot.n_faces} caras · rota: {rota.n_faces} caras "
          f"({faltante:.0%} de área faltante en 5 agujeros)")

    t0 = time.perf_counter()
    mesh, _ = conforming_tet_mesh(rota)
    print(f"CDT: {mesh.n_tets} tets en {time.perf_counter()-t0:.1f}s")

    resultados = {}
    for nombre, gamma in (("umbral", 0.0), ("graphcut", 40.0)):
        t0 = time.perf_counter()
        res = graphcut_segment(rota, mesh=mesh, gamma=gamma)
        seg = time.perf_counter() - t0

        recon = Mesh(res["V"], res["F"])
        d = cKDTree(res["V"]).query(spot.V)[0]
        area_rel = recon.total_area / spot.total_area

        resultados[nombre] = res
        print(f"  {nombre:9s} (γ={gamma:g}): {len(res['F'])} caras, "
              f"{connected_components(recon)} componentes, "
              f"área {area_rel:.3f}×, fidelidad {d.mean():.5f}, {seg:.1f}s")

    t0 = time.perf_counter()
    V_suave, F_suave = fair_filled_patches(resultados["graphcut"]["V"],
                                           resultados["graphcut"]["F"], rota)
    d = cKDTree(V_suave).query(spot.V)[0]
    print(f"  suavizado de las tapas: {len(F_suave)} caras, "
          f"área {Mesh(V_suave, F_suave).total_area / spot.total_area:.3f}×, "
          f"fidelidad {d.mean():.5f}, {time.perf_counter()-t0:.1f}s")

    columnas = [
        (rota.V, rota.F, ROJO, f"entrada — falta {faltante:.0%} en 5 agujeros"),
        (resultados["umbral"]["V"], resultados["umbral"]["F"], AZUL,
         "umbral simple (γ=0)\nEcuación 7"),
        (resultados["graphcut"]["V"], resultados["graphcut"]["F"], VERDE,
         "corte de grafo (γ=40)\nEcuación 8"),
        (V_suave, F_suave, VIOLETA, "corte de grafo + tapas suavizadas\nla entrada no se mueve"),
        (spot.V, spot.F, GRIS, "Spot original\n(referencia)"),
    ]
    caras = [cara_de_cerca(V, F, color) for V, F, color, _ in columnas]
    filas = 2 if all(c is not None for c in caras) else 1

    fig = plt.figure(figsize=(21.0, 4.6 * filas))
    ancho = 1.0 / len(columnas)
    alto = 1.0 / filas
    for k, (V, F, color, titulo) in enumerate(columnas):
        dibujar(fig.add_axes([k * ancho, 1.0 - alto, ancho, alto * 0.95], projection="3d"),
                V, F, color, titulo, limites)
        if filas == 2:
            ax = fig.add_axes([k * ancho + 0.01, 0.0, ancho - 0.02, alto * 0.92])
            ax.imshow(caras[k])
            ax.set_axis_off()
            ax.set_title("la cara, de cerca", fontsize=10)
    fig.suptitle("SPOT — segmentación robusta por corte de grafo (Jacobson, "
                 "Kavan y Sorkine-Hornung 2013)", fontsize=12.5, y=1.04,
                 fontweight="bold")

    destino = SALIDA / "segmentacion_spot.png"
    fig.savefig(destino, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigura en {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
