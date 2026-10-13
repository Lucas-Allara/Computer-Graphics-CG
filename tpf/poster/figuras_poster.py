"""Figuras del póster, sin texto: los rótulos los pone poster.tex.

Usa los resultados guardados por demos/demo_niveles_de_dano.py (correrlo antes).

    python poster/figuras_poster.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from gsd.mesh import Mesh  # noqa: E402

DATOS = RAIZ / "figuras" / "datos_niveles"
SALIDA = Path(__file__).resolve().parent / "figuras"
FUENTES = Path.home() / ".TinyTeX" / "texmf-dist" / "fonts" / "opentype" / "public" / "fira"

COLORES = {"entrada": "#c97f6f", "metodo": "#2a78d6", "suavizado": "#eb6834",
           "simetria": "#1baf7a", "original": "#9ea3a8"}
COLUMNAS = ("entrada", "metodo", "suavizado", "simetria", "original")
ETIQUETAS = {"metodo": "método de [1]", "suavizado": "con suavizado", "simetria": "con suavizado y simetría"}
TINTA = "#1d2433"
TINTA_SECUNDARIA = "#4a5160"
GRILLA = "#dcdcd6"
MM = 1 / 25.4


# ------------------------------------------------------------------ renders

def _poly(V, F):
    import pyvista as pv
    return pv.PolyData(np.asarray(V, float), np.hstack([np.full((len(F), 1), 3), F]).ravel())


def _render(capas, camara, tam, zoom=1.0, aristas=False, plano=None) -> Image.Image:
    """``capas``: lista de ``(V, F, color)``. Devuelve RGBA con fondo transparente."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=tam)
    p.set_background("white")
    for V, F, color in capas:
        p.add_mesh(_poly(V, F), color=color, specular=0.25, specular_power=20,
                   show_edges=aristas, edge_color="#2b2b2b", line_width=tam[0] / 700)
    if plano is not None:
        p.add_mesh(plano, color="#3d3d3d", opacity=0.22)
        borde = plano.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                            manifold_edges=False)
        p.add_mesh(borde, color="#3d3d3d", line_width=tam[0] / 300)
    p.camera_position = camara
    p.camera.zoom(zoom)
    p.enable_anti_aliasing("ssaa")
    img = Image.fromarray(p.screenshot(return_img=True, transparent_background=True))
    p.close()
    return img


def _recortar_juntas(imagenes, margen=12):
    """Recorta todas con la misma caja, la unión de sus contenidos."""
    cajas = [c for c in (im.getchannel("A").getbbox() for im in imagenes) if c]
    x0 = max(min(c[0] for c in cajas) - margen, 0)
    y0 = max(min(c[1] for c in cajas) - margen, 0)
    x1 = min(max(c[2] for c in cajas) + margen, imagenes[0].width)
    y1 = min(max(c[3] for c in cajas) + margen, imagenes[0].height)
    return [im.crop((x0, y0, x1, y1)) for im in imagenes]


def _fila(imagenes, destino: Path) -> None:
    """Pega las imágenes una al lado de la otra, todas del mismo ancho."""
    imagenes = _recortar_juntas(imagenes)
    fila = Image.new("RGBA", (sum(i.width for i in imagenes), imagenes[0].height), (0, 0, 0, 0))
    x = 0
    for im in imagenes:
        fila.paste(im, (x, 0))
        x += im.width
    fila.save(destino)
    print(f"{destino.relative_to(RAIZ)}: {fila.size}")


def _cargar(nombre: str, nivel: int) -> dict:
    d = np.load(DATOS / f"{nombre}_{nivel}.npz")
    v = np.load(DATOS / f"{nombre}_verdad.npz")
    mallas = {"entrada": (d["entrada_V"], d["entrada_F"]), "original": (v["V"], v["F"])}
    for k in ("metodo", "suavizado", "simetria"):
        mallas[k] = (d[f"{k}_V"], d[f"{k}_F"])
    return mallas


def _camara_al_agujero(mallas, arriba, dist, alto, foco=(0.0, 0.0, 0.0)):
    """Cámara horizontal del lado del agujero más grande."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    Ve, Fe = mallas["entrada"]
    Vo, Fo = mallas["original"]
    presentes = {tuple(sorted(map(tuple, q.tolist()))) for q in Ve[Fe]}
    falta = np.array([tuple(sorted(map(tuple, Vo[t].tolist()))) not in presentes for t in Fo])
    sub = Fo[falta]
    malla = Mesh(Vo, sub)
    fe = malla.face_edges.ravel()
    cara = np.repeat(np.arange(len(sub)), 3)
    orden = np.argsort(fe, kind="stable")
    fe, cara = fe[orden], cara[orden]
    pares = fe[1:] == fe[:-1]
    grafo = sp.coo_matrix((np.ones(pares.sum()), (cara[:-1][pares], cara[1:][pares])),
                          shape=(len(sub),) * 2)
    _, comp = connected_components(grafo, directed=False)
    grande = np.argmax(np.bincount(comp, weights=malla.face_areas))
    centro = Vo[sub[comp == grande]].reshape(-1, 3).mean(axis=0)
    arriba = np.asarray(arriba, float)
    h = centro - (centro @ arriba) * arriba
    h /= max(np.linalg.norm(h), 1e-9)
    foco = np.asarray(foco, float)
    return [tuple(foco + h * dist + arriba * alto), tuple(foco), tuple(arriba)]


def teaser() -> None:
    """Homer con 10% faltante, medio cuerpo."""
    m = _cargar("homer", 10)
    cam = [(0.85, 0.6, 2.4), (0.0, 0.2, 0.0), (0, 1, 0)]
    _fila([_render([(*m[k], COLORES[k])], cam, (2000, 2200), zoom=1.25) for k in COLUMNAS],
          SALIDA / "teaser.png")


def resultados() -> None:
    """Spot y Homer con 22% faltante, cuerpo entero."""
    for nombre, arriba, alto, tam in (("spot", (0, 0, 1), 1.0, (1800, 1620)),
                                      ("homer", (0, 1, 0), 0.6, (1440, 2160))):
        m = _cargar(nombre, 22)
        cam = _camara_al_agujero(m, arriba, dist=3.2, alto=alto)
        zoom = 1.05 if nombre == "spot" else 1.0
        _fila([_render([(*m[k], COLORES[k])], cam, tam, zoom=zoom) for k in COLUMNAS],
              SALIDA / f"{nombre}_22.png")


def tapa() -> None:
    """Primer plano del agujero del cuello de Spot: método y suavizado."""
    m = _cargar("spot", 22)
    cam = _camara_al_agujero(m, (0, 0, 1), dist=1.45, alto=0.4, foco=(-0.1, 0.0, 0.05))
    _fila([_render([(*m[k], COLORES[k])], cam, (1500, 1300), aristas=True)
           for k in ("metodo", "suavizado")], SALIDA / "tapa.png")


def reflejo() -> None:
    """Homer con 10% faltante: entrada, y entrada con el plano y las caras reflejadas."""
    import pyvista as pv

    from gsd.symmetry import mirror_completion, symmetry_plane

    m = _cargar("homer", 10)
    entrada = Mesh(*m["entrada"])
    n, d, _ = symmetry_plane(entrada)
    reflejadas = mirror_completion(entrada, Mesh(*m["suavizado"]), n, d)
    plano = pv.Plane(center=n * d + np.array([0, 0.33, 0.0]), direction=n, i_size=0.75, j_size=0.62)
    cam = [(1.25, 0.55, 2.1), (0.0, 0.3, 0.0), (0, 1, 0)]
    capas = [(*m["entrada"], COLORES["entrada"])]
    _fila([_render(capas, cam, (1600, 1440), zoom=1.35),
           _render(capas + [(reflejadas.V, reflejadas.F, COLORES["simetria"])], cam,
                   (1600, 1440), zoom=1.35, plano=plano)],
          SALIDA / "reflejo.png")


# --------------------------------------------------------------- matplotlib

def _estilo():
    import matplotlib
    from matplotlib import font_manager

    matplotlib.use("Agg")
    for archivo in ("FiraSans-Regular.otf", "FiraSans-Medium.otf", "FiraSans-SemiBold.otf"):
        font_manager.fontManager.addfont(str(FUENTES / archivo))
    matplotlib.rcParams.update({
        "font.family": "Fira Sans", "font.size": 22, "mathtext.fontset": "custom",
        "mathtext.rm": "Fira Sans", "mathtext.it": "Fira Sans:italic",
        "axes.edgecolor": GRILLA, "axes.labelcolor": TINTA_SECUNDARIA,
        "xtick.color": TINTA_SECUNDARIA, "ytick.color": TINTA_SECUNDARIA,
        "mathtext.cal": "Fira Sans", "text.color": TINTA, "pdf.fonttype": 42,
    })


def corte_w() -> None:
    """w en el plano y = 0 de una esfera cerrada y de la misma sin un casquete."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from gsd.shapes import icosphere, punch_hole
    from gsd.winding import sample_grid

    _estilo()
    mapa = LinearSegmentedColormap.from_list("w", ["#ffffff", "#bcd6f3", "#5d9be3"])
    sana = icosphere(4)
    rota = punch_hole(sana, np.array([0.0, 0.0, 1.0]), 0.6)
    t = np.linspace(0, 2 * np.pi, 2000)
    x, z = np.sin(t), np.cos(t)

    fig, ejes = plt.subplots(1, 2, figsize=(205 * MM, 96 * MM), constrained_layout=True)
    for ax, malla, hueco in ((ejes[0], sana, False), (ejes[1], rota, True)):
        w, ext = sample_grid(malla, resolution=320, padding=0.4, axis=1)
        im = ax.imshow(w, extent=ext, origin="lower", cmap=mapa, vmin=0, vmax=1,
                       interpolation="bilinear")
        ax.contour(w, levels=[0.5], extent=ext, colors=TINTA, linewidths=1.6, linestyles="--")
        borde = np.ones_like(t, bool) if not hueco else np.hypot(x, z - 1) >= 0.6
        xs, zs = np.where(borde, x, np.nan), np.where(borde, z, np.nan)
        ax.plot(xs, zs, color=TINTA, lw=3.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for lado in ax.spines.values():
            lado.set_visible(False)
    barra = fig.colorbar(im, ax=ejes, shrink=0.85, ticks=[0, 0.5, 1], pad=0.02)
    barra.ax.tick_params(labelsize=22, length=0)
    barra.ax.set_yticklabels(["0", "0,5", "1"])
    barra.outline.set_visible(False)
    barra.ax.set_title("$w$", fontsize=26, pad=10)
    destino = SALIDA / "corte_w.pdf"
    fig.savefig(destino, transparent=True)
    plt.close(fig)
    print(destino.relative_to(RAIZ))


def grafico() -> None:
    """Error en los agujeros contra superficie quitada, un panel por modelo."""
    import json

    import matplotlib.pyplot as plt

    _estilo()
    filas = json.loads((RAIZ / "figuras" / "niveles_de_dano.json").read_text())
    niveles = (0.05, 0.10, 0.22)
    variantes = ("metodo", "suavizado", "simetria")
    x = 100 * np.array(niveles)

    fig, ejes = plt.subplots(1, 2, figsize=(285 * MM, 138 * MM), sharey=True,
                             constrained_layout=True)
    for ax, nombre in zip(ejes, ("spot", "homer")):
        for k, v in enumerate(variantes):
            valores = np.array([[f[v] for f in filas if f["modelo"] == nombre and f["nivel"] == n]
                                for n in niveles])
            medio = valores.mean(axis=1)
            ax.vlines(x + 0.55 * (k - 1), valores.min(axis=1), valores.max(axis=1),
                      color=COLORES[v], lw=3, alpha=0.55)
            ax.plot(x, medio, color=COLORES[v], lw=4.5, marker="o", ms=13,
                    markeredgecolor="white", markeredgewidth=2.2, label=ETIQUETAS[v])
            ax.annotate(f"{medio[-1]:.1f}".replace(".", ","), (x[-1], medio[-1]), xytext=(14, 0),
                        textcoords="offset points", va="center", color=COLORES[v],
                        fontsize=22, fontweight="semibold")
        ax.set_title(nombre.capitalize(), loc="left", fontsize=28, fontweight="semibold",
                     color=TINTA)
        ax.set_xticks(x, [f"{int(n)} %" for n in x])
        ax.set_xlim(x[0] - 2, x[-1] + 6)
        ax.set_xlabel("superficie quitada", fontsize=22)
        ax.grid(axis="y", color=GRILLA, lw=1.2)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=21)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
    ejes[0].set_ylabel("error medio en los agujeros", fontsize=22)
    ejes[0].set_ylim(0, 4.8)
    ejes[0].yaxis.set_major_formatter(lambda v, _: f"{v:g} %".replace(".", ","))
    fig.legend(*ejes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3,
               frameon=False, fontsize=22, handlelength=1.6, columnspacing=1.6)
    destino = SALIDA / "error.pdf"
    fig.savefig(destino, transparent=True)
    plt.close(fig)
    print(destino.relative_to(RAIZ))


def main() -> None:
    SALIDA.mkdir(exist_ok=True)
    pedidas = sys.argv[1:] or ["teaser", "resultados", "tapa", "reflejo", "corte_w", "grafico"]
    for nombre in pedidas:
        globals()[nombre]()


if __name__ == "__main__":
    main()
