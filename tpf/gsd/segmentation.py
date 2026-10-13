"""Segmentación por corte de grafo (Sección 5 del paper).

Cada tetraedro de una tetraedrización conforme se etiqueta adentro o afuera
minimizando la energía de la Ecuación 8: un término de datos que sale de w y uno
de suavidad entre vecinos. La salida son las caras que separan las dos
etiquetas.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .mesh import Mesh
from .tetmesh import TetMesh
from .winding import WindingHierarchy, winding_number

# Regla simétrica de 4 puntos en el tetraedro (exacta hasta grado 2), en
# coordenadas baricéntricas. Todos pesan 1/4.
_A = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
_B = (5.0 - np.sqrt(5.0)) / 20.0
_BARICENTRICAS_4PT = np.array([
    [_A, _B, _B, _B],
    [_B, _A, _B, _B],
    [_B, _B, _A, _B],
    [_B, _B, _B, _A],
])


def tet_adjacency(mesh: TetMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pares de tetraedros que comparten una cara y el área de esa cara: el
    ``N(i)`` y el ``a_ij`` de la Ecuación 10."""
    tf = mesh.tet_faces.ravel()
    tet_id = np.repeat(np.arange(mesh.n_tets), 4)
    orden = np.argsort(tf, kind="stable")
    tet_ord = tet_id[orden]

    # Ordenado por cara, el bloque de la cara f empieza en la suma de los
    # conteos de las anteriores.
    inicio = np.concatenate([[0], np.cumsum(mesh.face_tet_count)])[:-1]
    interior = np.flatnonzero(mesh.face_tet_count == 2)

    tet_a = tet_ord[inicio[interior]]
    tet_b = tet_ord[inicio[interior] + 1]
    return tet_a, tet_b, mesh.face_areas[interior]


def element_winding_number(surf: Mesh, mesh: TetMesh, quadrature: int = 4,
                           hierarchical: bool = True) -> np.ndarray:
    """w promediado sobre cada tetraedro, en el baricentro (``quadrature=1``)
    o con la regla de 4 puntos."""
    V = mesh.V[mesh.T]
    evaluar = WindingHierarchy(surf) if hierarchical else (lambda P: winding_number(surf, P))

    if quadrature == 1:
        return evaluar(V.mean(axis=1))
    if quadrature != 4:
        raise ValueError("quadrature debe ser 1 o 4")

    puntos = np.einsum("qk,mkj->qmj", _BARICENTRICAS_4PT, V)
    return evaluar(puntos.reshape(-1, 3)).reshape(4, -1).mean(axis=0)


def data_term(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ecuación 9: costo de etiquetar cada elemento adentro y afuera."""
    return np.maximum(1.0 - w, 0.0), np.maximum(w, 0.0)


def smoothness_weights(w: np.ndarray, tet_a: np.ndarray, tet_b: np.ndarray,
                       area: np.ndarray, sigma: float) -> np.ndarray:
    """Ecuación 10: costo de separar dos vecinos. Es chico donde w ya salta
    entre ellos."""
    dif2 = (w[tet_a] - w[tet_b]) ** 2
    return area * np.exp(-dif2 / (2.0 * sigma ** 2))


def estimate_sigma(w: np.ndarray, tet_a: np.ndarray, tet_b: np.ndarray,
                   percentil: float = 85.0) -> float:
    """σ de la Ecuación 10, que el paper no fija.

    Se usa un percentil alto de ``|w_i - w_j|`` entre vecinos. La mediana no
    sirve: casi todos los pares están lejos de la superficie y valen lo mismo,
    así que sale un σ diminuto y el término de suavidad no hace nada.
    """
    p = float(np.percentile(np.abs(w[tet_a] - w[tet_b]), percentil))
    return max(p, 1e-6)


def _a_mascara(x: np.ndarray, m: int) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == bool:
        return x
    mascara = np.zeros(m, dtype=bool)
    mascara[x] = True
    return mascara


def apply_hard_constraints(u_adentro: np.ndarray, u_afuera: np.ndarray,
                           v_pesos: np.ndarray,
                           forzar_adentro: np.ndarray | None = None,
                           forzar_afuera: np.ndarray | None = None,
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Fija la etiqueta de algunos tetraedros (máscaras o índices) poniendo
    costo infinito al lado contrario.

    El infinito es mayor que la suma de todos los costos del grafo, así que
    ningún corte mínimo viola una restricción.

    La Sección 5.2 pide algo más débil (que al menos uno de los dos tetraedros
    de cada cara de entrada quede adentro), que no es submodular y el paper
    resuelve con una heurística. Fijar uno solo de los dos sí se resuelve
    exacto.
    """
    u_adentro = np.array(u_adentro, dtype=np.float64, copy=True)
    u_afuera = np.array(u_afuera, dtype=np.float64, copy=True)
    if forzar_adentro is None and forzar_afuera is None:
        return u_adentro, u_afuera

    m = len(u_adentro)
    fa = np.zeros(m, dtype=bool) if forzar_adentro is None else _a_mascara(forzar_adentro, m)
    ff = np.zeros(m, dtype=bool) if forzar_afuera is None else _a_mascara(forzar_afuera, m)
    if np.any(fa & ff):
        raise ValueError("un tetraedro no puede forzarse a la vez adentro y afuera")

    infinito = 1.0 + u_adentro.sum() + u_afuera.sum() + 2.0 * v_pesos.sum()
    u_afuera[fa] = infinito
    u_adentro[fa] = 0.0
    u_adentro[ff] = infinito
    u_afuera[ff] = 0.0
    return u_adentro, u_afuera


def hard_constraints_from_surface(surf: Mesh, mesh: TetMesh,
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Para cada triángulo de ``surf`` que es cara de ``mesh``, marca adentro el
    tetraedro del lado opuesto a la normal y afuera el del lado de la normal.

    Los vértices tienen que coincidir exactamente: en una rejilla el vecino más
    cercano siempre existe y daría caras falsas.

    Forzar afuera es incorrecto con piezas encastradas (una cara enterrada en
    otra pieza tiene volumen de los dos lados), por eso ``graphcut_segment`` usa
    solo ``forzar_adentro``.
    """
    from scipy.spatial import cKDTree

    d, ids = cKDTree(mesh.V).query(np.asarray(surf.V))
    diagonal = float(np.linalg.norm(mesh.V.max(axis=0) - mesh.V.min(axis=0)))
    coincide = d <= 1e-12 * diagonal
    tet_a, tet_b, _ = tet_adjacency(mesh)
    interior = np.flatnonzero(mesh.face_tet_count == 2)
    indice = {tuple(cara): pos for pos, cara in
              enumerate(np.sort(mesh.faces[interior], axis=1))}

    normales = surf.face_normals
    forzar_adentro = np.zeros(mesh.n_tets, dtype=bool)
    forzar_afuera = np.zeros(mesh.n_tets, dtype=bool)

    for k, tri in enumerate(np.asarray(surf.F)):
        if not coincide[tri].all():
            continue
        pos = indice.get(tuple(sorted(int(ids[v]) for v in tri)))
        if pos is None:
            continue

        a, b = tet_a[pos], tet_b[pos]
        centro_cara = mesh.V[mesh.faces[interior[pos]]].mean(axis=0)
        centro_a = mesh.V[mesh.T[a]].mean(axis=0)
        if np.dot(normales[k], centro_a - centro_cara) > 0:
            afuera_t, adentro_t = a, b
        else:
            afuera_t, adentro_t = b, a
        forzar_adentro[adentro_t] = True
        forzar_afuera[afuera_t] = True

    return forzar_adentro, forzar_afuera


def min_cut_labels(u_adentro: np.ndarray, u_afuera: np.ndarray,
                   tet_a: np.ndarray, tet_b: np.ndarray,
                   v_pesos: np.ndarray, scale: float = 2.0 ** 20) -> np.ndarray:
    """Minimiza la Ecuación 8 con un corte s-t mínimo. Devuelve ``True`` para
    los elementos que quedan adentro.

    Fuente = adentro, sumidero = afuera; ``S→i`` cuesta ``u_afuera(i)``,
    ``i→T`` cuesta ``u_adentro(i)`` y cada par de vecinos lleva una arista en
    cada sentido. ``maximum_flow`` pide capacidades enteras, así que se escalan.
    """
    from scipy.sparse.csgraph import maximum_flow

    m = len(u_adentro)
    S, T = m, m + 1
    n = m + 2

    costos = np.concatenate([u_adentro, u_afuera, v_pesos])
    pico = float(costos.max()) if len(costos) and costos.max() > 0 else 1.0
    k = scale / pico

    filas = np.concatenate([np.full(m, S), np.arange(m), tet_a, tet_b])
    cols = np.concatenate([np.arange(m), np.full(m, T), tet_b, tet_a])
    caps = np.concatenate([u_afuera, u_adentro, v_pesos, v_pesos])
    grafo = sp.csr_matrix((np.round(caps * k).astype(np.int64), (filas, cols)), shape=(n, n))
    resultado = maximum_flow(grafo, S, T)

    # Lo alcanzable desde la fuente en el residual queda adentro.
    residual = (grafo - resultado.flow).tocsr()
    residual.eliminate_zeros()

    alcanzables = np.zeros(n, dtype=bool)
    alcanzables[S] = True
    frontera = [S]
    while frontera:
        u = frontera.pop()
        vecinos = residual.indices[residual.indptr[u]:residual.indptr[u + 1]]
        pesos = residual.data[residual.indptr[u]:residual.indptr[u + 1]]
        for v, p in zip(vecinos, pesos):
            if p > 0 and not alcanzables[v]:
                alcanzables[v] = True
                frontera.append(v)

    return alcanzables[:m]


def graphcut_segment(surf: Mesh, mesh: TetMesh | None = None,
                     gamma: float = 1.0, sigma: float | None = None,
                     quadrature: int = 4,
                     forzar_adentro: np.ndarray | None = None,
                     forzar_afuera: np.ndarray | None = None,
                     restricciones_duras: bool = False,
                     heuristicas: bool = False,
                     **kwargs) -> dict:
    """El método completo.

    Si no se pasa ``mesh`` se arma con ``conforming_tet_mesh(surf, **kwargs)``.
    Con ``gamma=0`` da lo mismo que el umbral ``w > 0.5``. Con
    ``restricciones_duras=True`` se agregan las de
    ``hard_constraints_from_surface`` (solo adentro). Con ``heuristicas=True``
    se aplican después del corte las restricciones opcionales de la Sección 5.2
    (``gsd.restricciones``).

    Devuelve un dict con ``w``, ``adentro``, ``V``, ``F``, ``mesh`` y ``sigma``.
    """
    if mesh is None:
        from .conforming import conforming_tet_mesh

        mesh, _ = conforming_tet_mesh(surf, **kwargs)

    if restricciones_duras:
        fa_auto, _ = hard_constraints_from_surface(surf, mesh)
        forzar_adentro = fa_auto if forzar_adentro is None else (
            _a_mascara(forzar_adentro, mesh.n_tets) | fa_auto)

    w = element_winding_number(surf, mesh, quadrature=quadrature)
    tet_a, tet_b, area = tet_adjacency(mesh)

    u_adentro, u_afuera = data_term(w)
    if sigma is None:
        sigma = estimate_sigma(w, tet_a, tet_b)
    v_pesos = gamma * smoothness_weights(w, tet_a, tet_b, area, sigma)

    if forzar_adentro is not None or forzar_afuera is not None:
        u_adentro, u_afuera = apply_hard_constraints(
            u_adentro, u_afuera, v_pesos, forzar_adentro, forzar_afuera)

    adentro = min_cut_labels(u_adentro, u_afuera, tet_a, tet_b, v_pesos)
    if heuristicas:
        from .restricciones import caras_de_entrada, enforce_optional_constraints

        adentro = enforce_optional_constraints(mesh, caras_de_entrada(surf, mesh), w, adentro,
                                               u_adentro, u_afuera, tet_a, tet_b, v_pesos)
    V, F = extract_label_boundary(mesh, adentro)
    return {"w": w, "adentro": adentro, "V": V, "F": F, "mesh": mesh, "sigma": sigma}


def extract_label_boundary(mesh: TetMesh, adentro: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Caras con exactamente un tetraedro de adentro, orientadas hacia afuera y
    con los vértices compactados. Incluye las del borde del dominio, que con la
    CDT de la cápsula convexa son parte de la entrada."""
    tf = mesh.tet_faces.ravel()
    tet = np.repeat(np.arange(mesh.n_tets), 4)
    sel = adentro[tet]
    sel &= (np.bincount(tf[sel], minlength=mesh.n_faces) == 1)[tf]
    F = mesh.faces[tf[sel]].copy()

    centro_tet = mesh.V[mesh.T[tet[sel]]].mean(axis=1)
    centro_cara = mesh.V[F].mean(axis=1)
    normal = np.cross(mesh.V[F[:, 1]] - mesh.V[F[:, 0]], mesh.V[F[:, 2]] - mesh.V[F[:, 0]])
    hacia_afuera = np.einsum("ij,ij->i", normal, centro_cara - centro_tet) > 0
    F[~hacia_afuera] = F[~hacia_afuera][:, [0, 2, 1]]

    usados, F_compacta = np.unique(F, return_inverse=True)
    return mesh.V[usados], F_compacta.reshape(F.shape).astype(np.int64)
