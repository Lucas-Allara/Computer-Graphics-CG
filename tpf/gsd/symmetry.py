"""Completar los agujeros reflejando la entrada por su plano de simetría.

El suavizado no puede inventar detalle: si un agujero se lleva la nariz, la
tapa queda lisa. Si el modelo es simétrico, lo que falta de un lado suele estar
del otro.

1. Se busca el plano que minimiza la distancia de la superficie reflejada a la
   original, sin contar el 20% de puntos que peor coincide (los agujeros).
2. Se toman las caras reflejadas que caen donde la entrada no tiene superficie
   y que están conectadas con alguna cercana al relleno de una primera pasada.
   Lo segundo evita sumar piezas que no son simétricas en la pose (un brazo
   más arriba que el otro).
3. Se reconstruye de nuevo con esas caras agregadas a la entrada, sin coserlas.

Si el residuo del plano supera media arista, el modelo se toma como no
simétrico y no se hace nada.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh


def _muestras(V: np.ndarray, F: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """``k`` puntos uniformes sobre la superficie, por área."""
    T = V[F]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    f = rng.choice(len(F), size=k, p=area / area.sum())
    u, v = rng.random(k), rng.random(k)
    afuera = u + v > 1
    u[afuera], v[afuera] = 1 - u[afuera], 1 - v[afuera]
    return T[f, 0] + u[:, None] * (T[f, 1] - T[f, 0]) + v[:, None] * (T[f, 2] - T[f, 0])


def _reflejar(P: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """Reflexión de ``P`` respecto del plano ``n·x = d``."""
    return P - 2.0 * (P @ n - d)[:, None] * n


def _normal(tita: float, fi: float) -> np.ndarray:
    return np.array([np.cos(tita) * np.cos(fi), np.sin(tita) * np.cos(fi), np.sin(fi)])


def symmetry_plane(surf: Mesh, n_muestras: int = 4000, recorte: float = 0.8,
                   seed: int = 0) -> tuple[np.ndarray, float, float]:
    """Plano de simetría ``n·x = d`` de ``surf``. Devuelve ``(n, d, residuo)``,
    con el residuo la distancia media de la superficie reflejada a la original
    sobre la fracción ``recorte`` de puntos que mejor coinciden."""
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    V, F = np.asarray(surf.V), np.asarray(surf.F)
    P = _muestras(V, F, n_muestras, rng)
    arbol = cKDTree(_muestras(V, F, min(max(50_000, 10 * len(F)), 300_000), rng))
    corte = int(recorte * len(P))

    def costo(x: np.ndarray) -> float:
        d = arbol.query(_reflejar(P, _normal(x[0], x[1]), x[2]))[0]
        return float(np.mean(np.partition(d, corte)[:corte]))

    centro = P.mean(axis=0)
    _, _, ejes = np.linalg.svd(P - centro)
    mejor = None
    for n0 in ejes:
        x0 = np.array([np.arctan2(n0[1], n0[0]), np.arcsin(np.clip(n0[2], -1, 1)), centro @ n0])
        r = minimize(costo, x0, method="Nelder-Mead",
                     options=dict(xatol=1e-5, fatol=1e-7, maxiter=400))
        if mejor is None or r.fun < mejor.fun:
            mejor = r
    return _normal(mejor.x[0], mejor.x[1]), float(mejor.x[2]), float(mejor.fun)


def mirror_completion(surf: Mesh, relleno: Mesh, n: np.ndarray, d: float,
                      lejos: float = 2.0, cerca: float = 0.06, seed: int = 0) -> Mesh:
    """Caras de ``surf`` reflejadas que caen en sus agujeros.

    ``relleno`` es una primera reconstrucción. Se toman las caras reflejadas a
    más de ``lejos`` aristas de la entrada y conectadas (por caras así) con
    alguna a menos de ``cerca`` veces el lado de la caja del relleno. Reflejar
    invierte la orientación, así que se dan vuelta.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    h = float(np.median(surf.edge_lengths))
    lado = float((relleno.V.max(axis=0) - relleno.V.min(axis=0)).max())

    VR = _reflejar(np.asarray(surf.V), n, d)
    FR = np.asarray(surf.F)[:, ::-1]
    centros = VR[FR].mean(axis=1)
    entrada = cKDTree(_muestras(surf.V, surf.F, max(200_000, 20 * surf.n_faces), rng))
    en_el_vacio = entrada.query(centros)[0] > lejos * h
    hueco = cKDTree(_muestras(relleno.V, relleno.F, max(200_000, 20 * relleno.n_faces), rng))
    semilla = en_el_vacio & (hueco.query(centros)[0] < cerca * lado)

    # Componentes de las caras reflejadas en el vacío, unidas por aristas.
    reflejada = Mesh(VR, FR)
    fe = reflejada.face_edges[en_el_vacio].ravel()
    cara = np.repeat(np.flatnonzero(en_el_vacio), 3)
    orden = np.argsort(fe, kind="stable")
    fe, cara = fe[orden], cara[orden]
    pares = fe[1:] == fe[:-1]
    m = len(FR)
    grafo = sp.coo_matrix((np.ones(pares.sum()), (cara[:-1][pares], cara[1:][pares])), shape=(m, m))
    _, comp = connected_components(grafo, directed=False)
    quedan = en_el_vacio & np.isin(comp, np.unique(comp[semilla]))

    usados, F = np.unique(FR[quedan], return_inverse=True)
    return Mesh(VR[usados], F.reshape(-1, 3))


def reconstruct_with_symmetry(surf: Mesh, symmetry: bool = True, max_residuo: float = 0.5,
                              gamma: float = 40.0, quadrature: int = 1,
                              **kwargs) -> dict:
    """Segmentación, suavizado y, si el modelo es simétrico, una segunda
    pasada con la entrada completada por reflexión.

    ``**kwargs`` va a ``conforming_tet_mesh``. Devuelve un dict con ``metodo`` y
    ``suavizado`` (pares ``(V, F)`` de la primera pasada), ``plano``
    (``(n, d, residuo)``), ``reflejadas`` y ``simetria``; los últimos quedan en
    ``None`` si no se usó la simetría.
    """
    from .conforming import conforming_tet_mesh
    from .fairing import fair_filled_patches
    from .segmentation import graphcut_segment

    def pasada(entrada: Mesh) -> tuple[tuple, tuple]:
        mesh, _ = conforming_tet_mesh(entrada, **kwargs)
        r = graphcut_segment(entrada, mesh=mesh, gamma=gamma, quadrature=quadrature,
                             heuristicas=True)
        return (r["V"], r["F"]), fair_filled_patches(r["V"], r["F"], entrada)

    metodo, suavizado = pasada(surf)
    salida = {"metodo": metodo, "suavizado": suavizado, "plano": None, "simetria": None,
              "reflejadas": None}
    if not symmetry:
        return salida

    n, d, residuo = symmetry_plane(surf)
    salida["plano"] = (n, d, residuo)
    if residuo > max_residuo * float(np.median(surf.edge_lengths)):
        return salida

    reflejadas = mirror_completion(surf, Mesh(*suavizado), n, d)
    if reflejadas.n_faces == 0:
        return salida
    aumentada = Mesh(np.vstack([surf.V, reflejadas.V]),
                     np.vstack([surf.F, reflejadas.F + surf.n_vertices]))
    _, salida["simetria"] = pasada(aumentada)
    salida["reflejadas"] = reflejadas
    return salida
