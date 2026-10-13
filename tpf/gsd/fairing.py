"""Suavizado de las tapas que agrega la segmentación.

Donde había un agujero, la superficie de salida pasa por caras de la
tetraedrización y queda arrugada. Acá se dejan fijos los puntos que están sobre
la entrada y se reubican los de las tapas para que anulen el trilaplaciano
(``Δ³x = 0``), que empalma con lo que rodea al agujero en posición, tangente y
curvatura. Antes se arregla la conectividad de las tapas con flips de Delaunay
y una subdivisión.

Las caras de entrada no se tocan, y la malla de tetraedros y las etiquetas
tampoco: es un posproceso de la superficie.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .mesh import Mesh


def fair_filled_patches(V: np.ndarray, F: np.ndarray, surf: Mesh,
                        refine: int = 2, rounds: int = 3, order: int = 3
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Suaviza las tapas de ``(V, F)`` dejando fijo lo que está sobre ``surf``.

    ``refine`` es cuántas veces se subdividen las tapas (con tres ya empeora),
    ``rounds`` cuántas pasadas de suavizado y flips se hacen por nivel
    y ``order`` la potencia del laplaciano (2 es la placa delgada).
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    tol = 1e-9 * float(np.linalg.norm(surf.V.max(axis=0) - surf.V.min(axis=0)))

    fijo = _sobre_la_entrada(V, surf, tol)
    entrada = fijo[F].all(axis=1) & _sobre_la_entrada(V[F].mean(axis=1), surf, tol)
    if entrada.all():
        return V.copy(), F.copy()

    for nivel in range(refine + 1):
        if nivel > 0:
            V, F, entrada, fijo = _partir(Mesh(V, F), entrada, fijo)
        for _ in range(rounds):
            V = _suavizar(V, F, fijo, order)
            F = _flips(V, F, entrada)
            F, entrada, fijo = _sin_valencia_3(F, entrada, fijo)
            F, entrada, fijo = _colapsar_agujas(V, F, entrada, fijo)
    V = _suavizar(V, F, fijo, order)

    usados, F = np.unique(F, return_inverse=True)
    return V[usados], F.reshape(-1, 3)


def _partir(malla: Mesh, entrada: np.ndarray, fijo: np.ndarray):
    """Parte al medio todas las aristas de las tapas."""
    partir = np.ones(malla.n_edges, dtype=bool)
    partir[malla.face_edges[entrada].ravel()] = False
    fina = malla.split_edges(partir)
    fijo = np.concatenate([fijo, np.zeros(fina.n_vertices - malla.n_vertices, dtype=bool)])
    # split_edges pone primero, en orden, las caras que no se partieron, y
    # entre ellas están todas las de entrada.
    enteras = ~partir[malla.face_edges].any(axis=1)
    entrada = np.concatenate([entrada[enteras],
                              np.zeros(fina.n_faces - enteras.sum(), dtype=bool)])
    return fina.V, fina.F.astype(np.int64), entrada, fijo


def _sobre_la_entrada(P: np.ndarray, surf: Mesh, tol: float) -> np.ndarray:
    """Qué puntos de ``P`` están sobre algún triángulo de ``surf``.

    Incluye los puntos medios que agrega ``conforming_tet_mesh`` al subdividir
    caras que se cortan.
    """
    from scipy.spatial import cKDTree

    T = surf.V[surf.F]
    centros = T.mean(axis=1)
    radio = float(np.linalg.norm(T - centros[:, None], axis=2).max())
    cerca = cKDTree(centros).query_ball_point(P, radio + tol)
    i = np.repeat(np.arange(len(P)), [len(c) for c in cerca])
    j = np.concatenate([np.asarray(c, dtype=np.int64) for c in cerca]) if len(i) else i

    a, b, c = T[j, 0], T[j, 1], T[j, 2]
    n = np.cross(b - a, c - a)
    area2 = np.linalg.norm(n, axis=1)
    ok = area2 > 0
    n[ok] /= area2[ok, None]
    q = P[i]
    plano = np.abs(np.einsum("ij,ij->i", q - a, n)) <= tol
    tolb = tol * (np.linalg.norm(b - a, axis=1) + np.linalg.norm(c - a, axis=1))
    dentro = np.ones(len(i), dtype=bool)
    for u, v in ((a, b), (b, c), (c, a)):
        dentro &= np.einsum("ij,ij->i", np.cross(v - u, q - u), n) >= -tolb
    sobre = np.zeros(len(P), dtype=bool)
    sobre[i[ok & plano & dentro]] = True
    return sobre


# Cuánto puede salirse una tapa de la caja de sus vértices de anclaje, en
# fracciones de la diagonal de esa caja, antes de bajarle el orden.
_MARGEN = 0.15


def _suavizar(V: np.ndarray, F: np.ndarray, fijo: np.ndarray, orden: int) -> np.ndarray:
    """Resuelve ``L^orden x = 0`` en los vértices libres, con ``L`` el
    laplaciano uniforme y los fijos como borde.

    Con orden 2 o 3 una tapa puede dispararse lejos y dejar una púa. Cada tapa
    que se sale de la caja de sus anclajes (más ``_MARGEN``) se resuelve de
    nuevo con un orden menos; con orden 1 eso no pasa (principio del máximo).
    Las piezas libres que no tocan ningún fijo quedan donde están.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve

    e = Mesh(V, F).edges
    n = len(V)
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A

    fijo = fijo.copy()
    libre = ~fijo
    _, comp = connected_components(A[libre][:, libre], directed=False)
    toca_fijo = np.zeros(comp.max() + 1 if len(comp) else 0, dtype=bool)
    vecinos_fijos = np.asarray(A[libre][:, fijo].sum(axis=1)).ravel() > 0
    toca_fijo[comp[vecinos_fijos]] = True
    fijo[np.flatnonzero(libre)[~toca_fijo[comp]]] = True
    libre = ~fijo
    if not libre.any():
        return V.copy()

    idx = np.flatnonzero(libre)
    _, comp = connected_components(A[libre][:, libre], directed=False)
    A_lf = A[libre][:, fijo].tocsr()
    fijos_idx = np.flatnonzero(fijo)
    cajas = []
    for k in range(comp.max() + 1):
        anclas = fijos_idx[np.unique(A_lf[comp == k].indices)]
        lo, hi = V[anclas].min(axis=0), V[anclas].max(axis=0)
        margen = _MARGEN * float(np.linalg.norm(hi - lo))
        cajas.append((lo - margen, hi + margen))

    X = V.copy()
    pendientes = np.ones(comp.max() + 1, dtype=bool)
    for o in range(orden, 0, -1):
        quietos = ~libre
        quietos[idx[~pendientes[comp]]] = True  # las tapas ya resueltas no se mueven
        mover = ~quietos
        M = L
        for _ in range(o - 1):
            M = L @ M
        M = M.tocsr()
        X[mover] = spsolve(M[mover][:, mover].tocsc(), -(M[mover][:, quietos] @ X[quietos]))
        if o == 1:
            break
        for k in np.flatnonzero(pendientes):
            lo, hi = cajas[k]
            P = X[idx[comp == k]]
            pendientes[k] = bool(np.any((P < lo) | (P > hi)))
        if not pendientes.any():
            break
    return X


def _sin_valencia_3(F: np.ndarray, entrada: np.ndarray, fijo: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Saca los vértices libres de valencia 3 y deja el triángulo de sus
    vecinos. Al suavizar, sus tres caras se pliegan una sobre otra.

    El vértice sacado se marca fijo para que el sistema no quede singular.
    """
    malla = Mesh(np.zeros((F.max() + 1, 3)), F)
    valencia = np.bincount(malla.edges.ravel(), minlength=len(fijo))
    candidatos = np.flatnonzero((valencia == 3) & ~fijo)
    if not len(candidatos):
        return F, entrada, fijo

    existentes = {tuple(t) for t in np.sort(F, axis=1).tolist()}
    incidentes: dict[int, list[int]] = {}
    for f, cara in enumerate(F.tolist()):
        for v in cara:
            incidentes.setdefault(v, []).append(f)

    F = F.copy()
    fijo = fijo.copy()
    borrar = np.zeros(len(F), dtype=bool)
    for v in candidatos.tolist():
        caras = incidentes.get(v, [])
        if len(caras) != 3 or borrar[caras].any() or entrada[caras].any():
            continue
        # Cada cara, empezando en v, aporta la arista a -> b del anillo.
        siguiente = {}
        for f in caras:
            t = F[f].tolist()
            k = t.index(v)
            siguiente[t[(k + 1) % 3]] = t[(k + 2) % 3]
        a = next(iter(siguiente))
        b = siguiente.get(a)
        c = siguiente.get(b)
        if c is None or siguiente.get(c) != a:
            continue
        nueva = tuple(sorted((a, b, c)))
        if nueva in existentes:
            continue
        existentes.add(nueva)
        F[caras[0]] = [a, b, c]
        borrar[caras[1:]] = True
        fijo[v] = True
    return F[~borrar], entrada[~borrar], fijo


def _colapsar_agujas(V: np.ndarray, F: np.ndarray, entrada: np.ndarray,
                     fijo: np.ndarray, angulo_min: float = 10.0
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Colapsa la arista más corta de los triángulos de las tapas con un ángulo
    menor a ``angulo_min`` grados.

    Solo si los extremos comparten exactamente dos vecinos (condición de enlace),
    para que la superficie siga siendo una variedad.
    """
    malla = Mesh(V, F)
    agujas = np.flatnonzero(~entrada & (np.degrees(malla.corner_angles.min(axis=1)) < angulo_min))
    if not len(agujas):
        return F, entrada, fijo

    vecinos: dict[int, set[int]] = {}
    for a, b in malla.edges.tolist():
        vecinos.setdefault(a, set()).add(b)
        vecinos.setdefault(b, set()).add(a)
    incidentes: dict[int, list[int]] = {}
    for f, cara in enumerate(F.tolist()):
        for v in cara:
            incidentes.setdefault(v, []).append(f)

    F = F.copy()
    fijo = fijo.copy()
    borrar = np.zeros(len(F), dtype=bool)
    tocados: set[int] = set()
    for f in agujas.tolist():
        if borrar[f]:
            continue
        t = F[f]
        largos = [np.linalg.norm(V[t[(k + 2) % 3]] - V[t[(k + 1) % 3]]) for k in range(3)]
        k = int(np.argmin(largos))
        a, b = int(t[(k + 1) % 3]), int(t[(k + 2) % 3])
        if fijo[a] and fijo[b]:
            continue
        if fijo[a]:
            a, b = b, a  # a desaparece, así que tiene que ser libre
        if a in tocados or b in tocados or len(vecinos[a] & vecinos[b]) != 2:
            continue
        caras_a = [g for g in incidentes[a] if not borrar[g]]
        if entrada[caras_a].any():
            continue
        for g in caras_a:
            if b in F[g]:
                borrar[g] = True
            else:
                F[g][F[g] == a] = b
        fijo[a] = True
        tocados |= {a, b} | vecinos[a]
    return F[~borrar], entrada[~borrar], fijo


def _angulo(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """Ángulo en ``p`` del triángulo ``pqr``."""
    u, v = q - p, r - p
    return float(np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v))


def _flips(V: np.ndarray, F: np.ndarray, entrada: np.ndarray,
           max_pasadas: int = 50) -> np.ndarray:
    """Da vuelta las aristas de las tapas que no son localmente de Delaunay.

    No se hace el flip si la arista nueva ya existe o si algún triángulo nuevo
    queda dado vuelta.
    """
    F = F.copy()
    for _ in range(max_pasadas):
        malla = Mesh(V, F)
        existentes = {tuple(a) for a in malla.edges.tolist()}
        # Las dos caras de cada arista.
        fe = malla.face_edges.ravel()
        orden = np.argsort(fe, kind="stable")
        cara = orden // 3
        inicio = np.searchsorted(fe[orden], np.arange(malla.n_edges))
        candidatas = np.flatnonzero(malla.edge_face_count == 2)
        f1, f2 = cara[inicio[candidatas]], cara[inicio[candidatas] + 1]
        sirve = ~entrada[f1] & ~entrada[f2]

        tocadas: set[int] = set()
        hechos = 0
        for arista, g1, g2 in zip(candidatas[sirve], f1[sirve], f2[sirve]):
            if g1 in tocadas or g2 in tocadas:
                continue
            u, v = (int(x) for x in malla.edges[arista])
            t = list(F[g1])
            k = t.index(u)
            t = t[k:] + t[:k]
            if t[1] != v:  # g1 tiene que recorrer u -> v
                g1, g2 = g2, g1
                t = list(F[g1])
                k = t.index(u)
                t = t[k:] + t[:k]
            a, b, c = t
            d = next(x for x in F[g2] if x != a and x != b)
            if (min(c, d), max(c, d)) in existentes:
                continue
            if _angulo(V[c], V[a], V[b]) + _angulo(V[d], V[a], V[b]) <= np.pi:
                continue
            viejo = np.cross(V[b] - V[a], V[c] - V[a]) + np.cross(V[a] - V[b], V[d] - V[b])
            n1 = np.cross(V[d] - V[a], V[c] - V[a])
            n2 = np.cross(V[b] - V[d], V[c] - V[d])
            if n1 @ viejo <= 0 or n2 @ viejo <= 0:
                continue
            F[g1] = [a, d, c]
            F[g2] = [d, b, c]
            existentes.discard((min(a, b), max(a, b)))
            existentes.add((min(c, d), max(c, d)))
            tocadas |= {int(g1), int(g2)}
            hechos += 1
        if not hechos:
            break
    return F
