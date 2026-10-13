"""Tetraedrización con cada triángulo de entrada como cara (Sección 5 del
paper). Usa TetGen.

TetGen aborta el proceso si dos triángulos de la restricción se cortan. Por eso
esos triángulos se detectan antes y se subdividen; los pedazos que siguen
cortándose quedan fuera de la restricción (pero igual cuentan para w).

También están las dos alternativas sin restricción: Delaunay sobre los mismos
puntos y una rejilla regular.
"""

from __future__ import annotations

import numpy as np

from .mesh import Mesh
from .tetmesh import TetMesh, drop_degenerate_tets, tet_grid

# Caras de la caja envolvente, orientadas hacia afuera.
_CAJA_F = np.array([
    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
    [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
    [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
])


def _orient3d(a, b, c, d) -> np.ndarray:
    """Signo del volumen de ``abcd`` por filas, con 0 si es del orden del
    redondeo. Sin la tolerancia, caras vecinas (que comparten vértices)
    parecerían cortarse."""
    u, v, w = b - a, c - a, d - a
    vol = np.einsum("ij,ij->i", u, np.cross(v, w))
    escala = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) * np.linalg.norm(w, axis=1)
    return np.where(np.abs(vol) <= 1e-10 * escala, 0.0, np.sign(vol))


def _arista_atraviesa(P, Q, A, B, C) -> np.ndarray:
    """Si el segmento ``PQ`` atraviesa el triángulo ``ABC`` (incluido su borde)."""
    lados = _orient3d(A, B, C, P) * _orient3d(A, B, C, Q)
    t1, t2, t3 = _orient3d(P, Q, A, B), _orient3d(P, Q, B, C), _orient3d(P, Q, C, A)
    adentro = ((t1 >= 0) & (t2 >= 0) & (t3 >= 0)) | ((t1 <= 0) & (t2 <= 0) & (t3 <= 0))
    return (lados < 0) & adentro


def self_intersecting_faces(surf: Mesh) -> np.ndarray:
    """Máscara de los triángulos que cortan a algún otro.

    Dos triángulos no coplanares se cortan si una arista de uno atraviesa al
    otro. Los coplanares que se solapan no se detectan.
    """
    from scipy.spatial import cKDTree

    T = np.asarray(surf.V)[np.asarray(surf.F)]
    mascara = np.zeros(len(T), dtype=bool)
    if len(T) < 2:
        return mascara

    # Pares candidatos: centros a menos de la suma de los radios. Se buscan por
    # clases de radio para que un triángulo grande no vuelva cuadrática la
    # búsqueda.
    centros = T.mean(axis=1)
    radios = np.linalg.norm(T - centros[:, None], axis=2).max(axis=1)
    radios = np.maximum(radios, 1e-12 * radios.max())
    clase = np.floor(np.log2(radios / radios.min())).astype(np.int64)
    grupos = [np.flatnonzero(clase == k) for k in np.unique(clase)]
    arboles = [cKDTree(centros[g]) for g in grupos]
    r_max = [float(radios[g].max()) for g in grupos]

    pares = []
    for a in range(len(grupos)):
        for b in range(a, len(grupos)):
            d = arboles[a].sparse_distance_matrix(arboles[b], r_max[a] + r_max[b],
                                                  output_type="ndarray")
            i, j = grupos[a][d["i"]], grupos[b][d["j"]]
            par = np.column_stack([i, j])
            pares.append(par if a != b else par[i < j])
    pares = np.concatenate(pares)
    i, j = pares[:, 0], pares[:, 1]

    lo, hi = T.min(axis=1), T.max(axis=1)
    solapan = np.all((lo[i] <= hi[j]) & (lo[j] <= hi[i]), axis=1)
    i, j = i[solapan], j[solapan]

    cortan = np.zeros(len(i), dtype=bool)
    for k in range(3):
        s = (k + 1) % 3
        cortan |= _arista_atraviesa(T[i, k], T[i, s], T[j, 0], T[j, 1], T[j, 2])
        cortan |= _arista_atraviesa(T[j, k], T[j, s], T[i, 0], T[i, 1], T[i, 2])

    mascara[i[cortan]] = True
    mascara[j[cortan]] = True
    return mascara


def _refinar_rojo_verde(V: np.ndarray, F: np.ndarray, marcadas: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Parte en cuatro las caras marcadas sin dejar vértices colgados: las
    vecinas con una arista partida se bisectan, y las que quedan con dos o más
    se parten en cuatro también."""
    malla = Mesh(V, F)
    marcadas = marcadas.copy()
    while True:
        partida = np.zeros(malla.n_edges, dtype=bool)
        partida[malla.face_edges[marcadas].ravel()] = True
        nuevas = ~marcadas & (partida[malla.face_edges].sum(axis=1) >= 2)
        if not nuevas.any():
            break
        marcadas |= nuevas
    fina = malla.split_edges(partida)
    return fina.V, fina.F.astype(np.int64)


def _caja(surf: Mesh, margin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Esquinas de la caja envolvente agrandada en ``margin`` veces su lado
    mayor, y sus extremos."""
    lo, hi = surf.bounding_box()
    aire = margin * float((hi - lo).max())
    lo, hi = lo - aire, hi + aire
    esquinas = np.array([[x, y, z] for x in (lo[0], hi[0])
                         for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return esquinas, lo, hi


def conforming_tet_mesh(surf: Mesh, max_far_edge: float | None = None, margin: float = 0.22,
                        quality: float = 1.6, refine_intersections: int = 2
                        ) -> tuple[TetMesh, np.ndarray]:
    """Tetraedrización con los triángulos de ``surf`` como caras.

    Devuelve ``(malla, ids)``, con ``ids[k]`` el vértice de la malla que
    corresponde al vértice ``k`` de la entrada.

    Por omisión es la del paper: la CDT de la cápsula convexa de la entrada,
    sin puntos nuevos. Si se da ``max_far_edge``, se tetraedriza la caja
    agrandada en ``margin`` y TetGen agrega puntos para acotar la calidad y el
    largo de las aristas lejos de la superficie. ``refine_intersections`` es
    cuántas veces se subdividen las caras que se cortan; si TetGen rechaza el
    resultado se prueba con una vez menos.
    """
    import tetgen
    from scipy.spatial import cKDTree

    if max_far_edge is None:
        # p: PLC, YY: sin puntos nuevos sobre las caras de entrada, c: conserva la
        # cápsula convexa.
        esquinas, cajas, switches = np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), "pYYcQ"
    else:
        esquinas, _, _ = _caja(surf, margin)
        cajas = _CAJA_F
        # q: calidad, a: volumen máximo.
        switches = f"pq{quality}YYa{max_far_edge ** 3 / 6.0:.12f}Q"

    # Las caras que no se cortan van a la restricción. Subdividir no cambia los
    # índices de los vértices de entrada (los nuevos van al final).
    V, F = np.asarray(surf.V), np.asarray(surf.F)
    cortan = self_intersecting_faces(surf)
    intentos = [(V, F[~cortan])]
    for _ in range(refine_intersections):
        if not cortan.any():
            break
        V, F = _refinar_rojo_verde(V, F, cortan)
        cortan = self_intersecting_faces(Mesh(V, F))
        intentos.append((V, F[~cortan]))

    for k, (V, restringidas) in enumerate(reversed(intentos)):
        try:
            salida = tetgen.TetGen(np.vstack([esquinas, V]),
                                   np.vstack([cajas, restringidas + len(esquinas)])
                                   ).tetrahedralize(switches=switches)
            break
        except RuntimeError:
            if k == len(intentos) - 1:
                raise
    malla = TetMesh(*drop_degenerate_tets(np.asarray(salida[0]), np.asarray(salida[1])))
    _, ids = cKDTree(malla.V).query(np.asarray(surf.V))
    return malla, ids.astype(np.int64)


def delaunay_tet_mesh(surf: Mesh, margin: float = 0.22) -> TetMesh:
    """Delaunay de los vértices de entrada y las esquinas de la caja, sin pedir
    que los triángulos de entrada sean caras."""
    from scipy.spatial import Delaunay

    esquinas, _, _ = _caja(surf, margin)
    puntos = np.vstack([esquinas, surf.V])
    return TetMesh(*drop_degenerate_tets(puntos, Delaunay(puntos).simplices))


def grid_tet_mesh(surf: Mesh, n: int = 32, margin: float = 0.22) -> TetMesh:
    """Rejilla regular de ``n`` cubos por lado que cubre la caja de ``surf``."""
    _, lo, hi = _caja(surf, margin)
    return tet_grid(n, size=float((hi - lo).max()), center=0.5 * (lo + hi))
