"""Completación por simetría. Las formas de prueba son esferas deformadas,
simétricas respecto de x = 0."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from gsd.mesh import Mesh
from gsd.shapes import icosphere, punch_hole
from gsd.symmetry import _muestras, mirror_completion, reconstruct_with_symmetry, symmetry_plane


def _con_chichones() -> Mesh:
    """Esfera estirada con dos chichones, simétrica solo respecto de x = 0."""
    esfera = icosphere(3)
    V = esfera.V * np.array([1.0, 0.8, 1.2])
    for lado in (-1.0, 1.0):
        centro = np.array([lado * 0.55, 0.3, 0.75])
        V = V + 0.35 * np.exp(-np.sum((V - centro) ** 2, axis=1) / 0.04)[:, None] * V
    return Mesh(V, esfera.F)


def _rotacion(seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q * np.sign(np.diag(r))


def test_el_plano_se_encuentra_aunque_la_forma_este_girada():
    forma = _con_chichones()
    R, t = _rotacion(4), np.array([0.3, -0.2, 0.1])
    girada = Mesh(forma.V @ R.T + t, forma.F)

    n, d, residuo = symmetry_plane(girada)
    verdadera = R @ np.array([1.0, 0.0, 0.0])
    angulo = np.degrees(np.arccos(min(1.0, abs(n @ verdadera))))
    assert angulo < 1.0
    assert abs(abs(d) - abs(verdadera @ t)) < 1e-2
    assert residuo < 0.5 * np.median(girada.edge_lengths)


def test_un_modelo_sin_simetria_se_reconoce():
    """Lucy no es simétrica: el residuo supera media arista.

    Una esfera deformada no sirve para esto: descartando el 20% que peor
    coincide, siempre encuentra un plano que casi anda."""
    from pathlib import Path

    from gsd.mesh import load_obj

    ruta = Path(__file__).resolve().parent.parent / "assets" / "lucy.obj"
    if not ruta.exists():
        pytest.skip("falta assets/lucy.obj")
    lucy = load_obj(ruta)
    _, _, residuo = symmetry_plane(lucy)
    assert residuo > 0.5 * np.median(lucy.edge_lengths)


def _con_cuernos() -> Mesh:
    """Como ``_con_chichones`` pero más fina y con salientes más marcadas."""
    esfera = icosphere(4)
    V = esfera.V * np.array([1.0, 0.8, 1.2])
    for lado in (-1.0, 1.0):
        centro = np.array([lado * 0.55, 0.3, 0.75])
        V = V + 0.6 * np.exp(-np.sum((V - centro) ** 2, axis=1) / 0.03)[:, None] * V
    return Mesh(V, esfera.F)


def test_lo_que_falta_de_un_lado_vuelve_desde_el_otro():
    """Sin la punta de una saliente, reflejar la otra deja la reconstrucción
    más cerca de la forma original que el suavizado solo."""
    pytest.importorskip("tetgen")
    forma = _con_cuernos()
    punta = forma.V[np.argmax(forma.V @ np.array([0.55, 0.3, 0.75]))]
    rota = punch_hole(forma, punta, 0.4)
    usados, F = np.unique(rota.F, return_inverse=True)
    rota = Mesh(rota.V[usados], F.reshape(-1, 3))

    salida = reconstruct_with_symmetry(rota)
    assert salida["simetria"] is not None
    assert salida["reflejadas"].n_faces > 0

    # Las caras de la forma que faltan en la entrada.
    entrada = {tuple(sorted(map(tuple, p.tolist()))) for p in rota.V[rota.F]}
    falta = np.array([tuple(sorted(map(tuple, forma.V[t].tolist()))) not in entrada
                      for t in forma.F])
    rng = np.random.default_rng(0)
    verdad = _muestras(forma.V, forma.F[falta], 20_000, rng)

    def error(VF):
        return cKDTree(_muestras(VF[0], VF[1], 200_000, rng)).query(verdad)[0].mean()

    assert error(salida["simetria"]) < 0.6 * error(salida["suavizado"])


def test_no_se_refleja_lo_que_ya_esta():
    """Sin agujeros no se refleja nada."""
    forma = _con_chichones()
    n, d, _ = symmetry_plane(forma)
    assert mirror_completion(forma, forma, n, d).n_faces == 0
