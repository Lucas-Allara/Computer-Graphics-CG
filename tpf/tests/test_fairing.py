"""Suavizado de las tapas: la entrada no se mueve y las tapas se acercan
a la esfera de la que salieron."""

import numpy as np
import pytest

from gsd.fairing import fair_filled_patches
from gsd.mesh import Mesh
from gsd.segmentation import graphcut_segment
from gsd.shapes import icosphere, random_holes

pytest.importorskip("tetgen", reason="la segmentación de partida necesita TetGen")


def _esfera_rota(subdivisiones: int) -> Mesh:
    """Esfera con agujeros, sin los vértices sueltos que deja ``random_holes``."""
    rota = random_holes(icosphere(subdivisiones), target_fraction=0.15, n_holes=4, seed=7)
    usados, F = np.unique(rota.F, return_inverse=True)
    return Mesh(rota.V[usados], F.reshape(-1, 3))


def _desvio_de_la_esfera(V: np.ndarray, F: np.ndarray) -> float:
    """Distancia media a la esfera unitaria, pesada por área."""
    T = V[F]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    P = np.concatenate([T.mean(axis=1), 0.5 * (T[:, 0] + T[:, 1]),
                        0.5 * (T[:, 1] + T[:, 2]), 0.5 * (T[:, 2] + T[:, 0])])
    peso = np.tile(area, 4)
    return float((np.abs(np.linalg.norm(P, axis=1) - 1.0) * peso).sum() / peso.sum())


@pytest.fixture(scope="module")
def caso():
    rota = _esfera_rota(3)
    resultado = graphcut_segment(rota, gamma=3.0)
    V, F = fair_filled_patches(resultado["V"], resultado["F"], rota)
    return rota, resultado, V, F


def test_la_entrada_no_se_mueve(caso):
    """Los vértices y caras de entrada que estaban en la salida siguen igual."""
    rota, resultado, V, F = caso
    vertice_de_entrada = {tuple(p) for p in rota.V.tolist()}
    antes = {tuple(p) for p in resultado["V"].tolist()} & vertice_de_entrada
    despues = {tuple(p) for p in V.tolist()} & vertice_de_entrada
    assert antes == despues

    caras_de_entrada = {frozenset(map(tuple, rota.V[f].tolist())) for f in rota.F}
    def presentes(V, F):
        return {c for c in (frozenset(map(tuple, V[f].tolist())) for f in F)
                if c in caras_de_entrada}
    assert presentes(V, F) == presentes(resultado["V"], resultado["F"])


def test_el_suavizado_no_abre_la_superficie(caso):
    """No aparecen bordes ni aristas no variedad nuevas."""
    _, resultado, V, F = caso
    antes = Mesh(resultado["V"], resultado["F"]).edge_face_count
    despues = Mesh(V, F).edge_face_count
    assert np.all(despues % 2 == 0)
    assert (despues > 2).sum() <= (antes > 2).sum()


def test_las_tapas_se_acercan_a_la_superficie_verdadera(caso):
    _, resultado, V, F = caso
    antes = _desvio_de_la_esfera(resultado["V"], resultado["F"])
    despues = _desvio_de_la_esfera(V, F)
    assert despues < 0.5 * antes


def test_sin_tapas_no_cambia_nada():
    sana = icosphere(2)
    resultado = graphcut_segment(sana, gamma=3.0)
    V, F = fair_filled_patches(resultado["V"], resultado["F"], sana)
    assert np.array_equal(V, resultado["V"])
    assert np.array_equal(F, resultado["F"])


def test_la_tapa_no_se_pliega():
    """Sobre una tetraedrización refinada, donde el método no deja pliegues,
    ningún par de caras vecinas queda doblado más de 120°."""
    rota = _esfera_rota(3)
    resultado = graphcut_segment(rota, gamma=3.0, max_far_edge=0.35)
    V, F = fair_filled_patches(resultado["V"], resultado["F"], rota, refine=1)
    malla = Mesh(V, F)
    fe = malla.face_edges.ravel()
    orden = np.argsort(fe, kind="stable")
    cara = orden // 3
    inicio = np.searchsorted(fe[orden], np.flatnonzero(malla.edge_face_count == 2))
    n = malla.face_normals
    coseno = np.einsum("ij,ij->i", n[cara[inicio]], n[cara[inicio + 1]])
    assert (coseno < np.cos(np.radians(120))).sum() == 0


def test_el_suavizado_no_deja_puas():
    """Sobre Homer con agujeros, nada se sale de la caja del modelo real más
    que un poco."""
    from pathlib import Path

    from gsd.conforming import conforming_tet_mesh
    from gsd.mesh import load_obj

    ruta = Path(__file__).resolve().parent.parent / "assets" / "homer.obj"
    if not ruta.exists():
        pytest.skip("falta assets/homer.obj")
    homer = load_obj(ruta)
    lo, hi = homer.V.min(axis=0), homer.V.max(axis=0)
    verdad = Mesh((homer.V - (lo + hi) / 2) * (1.35 / float((hi - lo).max())), homer.F)

    rota = random_holes(verdad, target_fraction=0.22, n_holes=5, seed=3)
    usados, F = np.unique(rota.F, return_inverse=True)
    rota = Mesh(rota.V[usados], F.reshape(-1, 3))
    mesh, _ = conforming_tet_mesh(rota)
    resultado = graphcut_segment(rota, mesh=mesh, gamma=40.0, quadrature=1)
    V, _ = fair_filled_patches(resultado["V"], resultado["F"], rota)

    lo, hi = verdad.V.min(axis=0), verdad.V.max(axis=0)
    afuera = np.maximum(V - hi, lo - V).max()
    assert afuera < 0.03
