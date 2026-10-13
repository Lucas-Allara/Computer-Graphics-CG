"""Error en los agujeros del método, el método con suavizado y el método con
simetría, sobre Spot y Homer con 5%, 10% y 22% de superficie faltante (tres
juegos de agujeros por nivel).

También mide, sobre la misma salida del método, el suavizado con el laplaciano
(como TetWild), con Δ² y con tres subdivisiones en vez de dos.

Escribe en ``figuras/``: ``error_vs_dano.png``, ``niveles_de_dano_spot.png``,
``niveles_de_dano_homer.png`` y los números en ``niveles_de_dano.json``.

    python demos/demo_niveles_de_dano.py            # unos 10 minutos
    python demos/demo_niveles_de_dano.py --figuras  # solo redibuja lo guardado
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsd.fairing import fair_filled_patches
from gsd.mesh import Mesh, load_obj
from gsd.shapes import random_holes
from gsd.symmetry import _muestras, reconstruct_with_symmetry

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "figuras"
DATOS = SALIDA / "datos_niveles"

MODELOS = {
    "spot": (RAIZ / "assets" / "spot" / "spot_triangulated.obj", True),
    "homer": (RAIZ / "assets" / "homer.obj", False),
}
NIVELES = (0.05, 0.10, 0.22)
SEMILLAS = (3, 7, 21)
SEMILLA_FIGURA = 3
TAMANO = 1.35

VARIANTES = ("metodo", "suavizado", "simetria")
ALTERNATIVAS = {"laplaciano": {"order": 1}, "bilaplaciano": {"order": 2}, "tres_subdivisiones": {"refine": 3}}
ETIQUETAS = {"metodo": "método del paper", "suavizado": "+ suavizado",
             "simetria": "+ simetría"}
COLORES = {"metodo": "#2a78d6", "suavizado": "#eb6834", "simetria": "#1baf7a"}
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
GRILLA = "#e4e3df"


def cargar(nombre: str) -> Mesh:
    ruta, permutar = MODELOS[nombre]
    m = load_obj(ruta)
    lo, hi = m.bounding_box()
    V = (m.V - (lo + hi) / 2) * (TAMANO / float((hi - lo).max()))
    if permutar:  # Spot trae el eje vertical en y
        V = np.column_stack([V[:, 0], -V[:, 2], V[:, 1]])
    return Mesh(V, m.F)


def agujerear(verdad: Mesh, fraccion: float, semilla: int) -> tuple[Mesh, np.ndarray]:
    """Entrada con agujeros, sin vértices sueltos, y máscara de las caras que
    faltan."""
    rota = random_holes(verdad, target_fraction=fraccion, n_holes=5, seed=semilla)
    usados, F = np.unique(rota.F, return_inverse=True)
    rota = Mesh(rota.V[usados], F.reshape(-1, 3))
    presentes = {tuple(sorted(map(tuple, p.tolist()))) for p in rota.V[rota.F]}
    falta = np.array([tuple(sorted(map(tuple, verdad.V[t].tolist()))) not in presentes
                      for t in verdad.F])
    return rota, falta


def error_en_agujeros(verdad_faltante: np.ndarray, V: np.ndarray, F: np.ndarray,
                      rng: np.random.Generator) -> float:
    """Distancia media de la superficie verdadera que falta a la reconstrucción,
    en porcentaje del tamaño del modelo."""
    arbol = cKDTree(_muestras(V, F, 300_000, rng))
    return 100.0 * float(arbol.query(verdad_faltante)[0].mean()) / TAMANO


def correr() -> list[dict]:
    DATOS.mkdir(parents=True, exist_ok=True)
    filas = []
    for nombre in MODELOS:
        verdad = cargar(nombre)
        for fraccion in NIVELES:
            for semilla in SEMILLAS:
                rng = np.random.default_rng(0)
                rota, falta = agujerear(verdad, fraccion, semilla)
                faltante = _muestras(verdad.V, verdad.F[falta], 100_000, rng)
                t0 = time.perf_counter()
                salida = reconstruct_with_symmetry(rota)
                segundos = time.perf_counter() - t0
                fila = {"modelo": nombre, "nivel": fraccion, "semilla": semilla,
                        "falta": 1.0 - rota.total_area / verdad.total_area,
                        "segundos": segundos}
                for v in VARIANTES:
                    VF = salida[v] if salida[v] is not None else salida["suavizado"]
                    fila[v] = error_en_agujeros(faltante, VF[0], VF[1], rng)
                for clave, opciones in ALTERNATIVAS.items():
                    VF = fair_filled_patches(*salida["metodo"], rota, **opciones)
                    fila[clave] = error_en_agujeros(faltante, VF[0], VF[1], rng)
                fila["reflejadas"] = 0 if salida["reflejadas"] is None else salida["reflejadas"].n_faces
                filas.append(fila)
                print(f"{nombre} {fraccion:.0%} semilla {semilla:2d}: "
                      + "  ".join(f"{v} {fila[v]:.3f}%" for v in VARIANTES)
                      + f"  ({segundos:.0f}s)", flush=True)
                if semilla == SEMILLA_FIGURA:
                    np.savez(DATOS / f"{nombre}_{int(fraccion * 100)}.npz",
                             entrada_V=rota.V, entrada_F=rota.F,
                             **{f"{v}_V": (salida[v] or salida["suavizado"])[0] for v in VARIANTES},
                             **{f"{v}_F": (salida[v] or salida["suavizado"])[1] for v in VARIANTES})
        np.savez(DATOS / f"{nombre}_verdad.npz", V=verdad.V, F=verdad.F)
    (SALIDA / "niveles_de_dano.json").write_text(json.dumps(filas, indent=1))
    return filas


def grafico(filas: list[dict]) -> None:
    """Error contra porcentaje faltante: un panel por modelo, la misma escala."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 14, "axes.edgecolor": GRILLA,
                         "axes.labelcolor": TINTA_SECUNDARIA,
                         "xtick.color": TINTA_SECUNDARIA, "ytick.color": TINTA_SECUNDARIA})
    fig, ejes = plt.subplots(1, len(MODELOS), figsize=(12, 4.8), sharey=True,
                             constrained_layout=True)
    x = 100 * np.array(NIVELES)
    for ax, nombre in zip(ejes, MODELOS):
        for k, v in enumerate(VARIANTES):
            valores = np.array([[f[v] for f in filas if f["modelo"] == nombre and f["nivel"] == n]
                                for n in NIVELES])
            medio = valores.mean(axis=1)
            # Rango entre juegos de agujeros, corrido para que no se pisen.
            ax.vlines(x + 0.45 * (k - 1), valores.min(axis=1), valores.max(axis=1),
                      color=COLORES[v], lw=1.5, alpha=0.7)
            ax.plot(x, medio, color=COLORES[v], lw=2.5, marker="o", ms=8,
                    markeredgecolor="white", markeredgewidth=1.5, label=ETIQUETAS[v])
            ax.annotate(f"{medio[-1]:.2f}%", (x[-1], medio[-1]), xytext=(10, 0),
                        textcoords="offset points", va="center", color=TINTA, fontsize=12)
        ax.set_title(nombre.capitalize(), color=TINTA, fontsize=16, loc="left")
        ax.set_xticks(x, [f"{int(n)}%" for n in x])
        ax.set_xlim(x[0] - 1.5, x[-1] + 5)
        ax.set_xlabel("superficie faltante")
        ax.grid(axis="y", color=GRILLA, lw=1)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    ejes[0].set_ylabel("error medio en los agujeros\n(% del tamaño del modelo)")
    ejes[0].set_ylim(bottom=0)
    fig.legend(*ejes[0].get_legend_handles_labels(), loc="outside upper center",
               ncol=3, frameon=False, fontsize=13)
    destino = SALIDA / "error_vs_dano.png"
    fig.savefig(destino, dpi=300)
    plt.close(fig)
    print(f"figura en {destino.relative_to(RAIZ)}")


def grillas() -> None:
    """Entrada, variantes y original, una fila por nivel. Cada fila se mira
    desde el lado del agujero más grande."""
    import pyvista as pv
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    pv.OFF_SCREEN = True

    def poly(V, F):
        return pv.PolyData(V, np.hstack([np.full((len(F), 1), 3), F]).ravel())

    for nombre in MODELOS:
        arriba = np.array([0.0, 0.0, 1.0]) if MODELOS[nombre][1] else np.array([0.0, 1.0, 0.0])
        v = np.load(DATOS / f"{nombre}_verdad.npz")
        verdad = Mesh(v["V"], v["F"])
        columnas = [("entrada", "entrada", "#c97f6f")] + \
                   [(v_, ETIQUETAS[v_], COLORES[v_]) for v_ in VARIANTES] + \
                   [("original", "original", "#9ea3a8")]
        alto = 1650 if nombre == "homer" else 1350
        p = pv.Plotter(off_screen=True, shape=(len(NIVELES), len(columnas)),
                       window_size=(380 * len(columnas), alto), border=False)
        for fila, fraccion in enumerate(NIVELES):
            d = np.load(DATOS / f"{nombre}_{int(fraccion * 100)}.npz")
            presentes = {tuple(sorted(map(tuple, q.tolist())))
                         for q in d["entrada_V"][d["entrada_F"]]}
            falta = np.array([tuple(sorted(map(tuple, verdad.V[t].tolist()))) not in presentes
                              for t in verdad.F])
            sub = verdad.F[falta]
            fe = Mesh(verdad.V, sub).face_edges.ravel()
            cara = np.repeat(np.arange(len(sub)), 3)
            orden = np.argsort(fe, kind="stable")
            fe, cara = fe[orden], cara[orden]
            pares = fe[1:] == fe[:-1]
            grafo = sp.coo_matrix((np.ones(pares.sum()), (cara[:-1][pares], cara[1:][pares])),
                                  shape=(len(sub),) * 2)
            _, comp = connected_components(grafo, directed=False)
            grande = np.argmax(np.bincount(comp, weights=Mesh(verdad.V, sub).face_areas))
            centro = verdad.V[sub[comp == grande]].reshape(-1, 3).mean(axis=0)
            horizontal = centro - (centro @ arriba) * arriba
            horizontal /= max(np.linalg.norm(horizontal), 1e-9)
            camara = [tuple(horizontal * 2.9 + arriba * 0.9), (0, 0, 0), tuple(arriba)]
            for col, (clave, titulo, color) in enumerate(columnas):
                if clave == "original":
                    V, F = verdad.V, verdad.F
                else:
                    V, F = d[f"{clave}_V"], d[f"{clave}_F"]
                p.subplot(fila, col)
                p.add_mesh(poly(V, F), color=color, specular=0.2)
                if fila == 0:
                    p.add_text(titulo, font_size=12, color=TINTA)
                if col == 0:
                    p.add_text(f"\n\nfalta {int(fraccion * 100)}%", font_size=11, color=TINTA)
                p.camera_position = camara
        destino = SALIDA / f"niveles_de_dano_{nombre}.png"
        p.screenshot(destino)
        p.close()
        print(f"figura en {destino.relative_to(RAIZ)}")


def main() -> None:
    if "--figuras" in sys.argv:
        filas = json.loads((SALIDA / "niveles_de_dano.json").read_text())
    else:
        filas = correr()
    grafico(filas)
    grillas()


if __name__ == "__main__":
    main()
