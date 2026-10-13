"""Número de vueltas generalizado (Jacobson, Kavan y Sorkine-Hornung, 2013).

``winding_number`` suma el ángulo sólido de todas las caras; ``WindingHierarchy``
es la evaluación jerárquica de la Sección 4.3, que da lo mismo sumando muchos
menos triángulos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh

_INV_4PI = 1.0 / (4.0 * np.pi)

# Elementos por bloque al evaluar muchos puntos, para acotar la memoria.
_CHUNK_ELEMS = 4_000_000


def solid_angles(V: np.ndarray, F: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Ángulo sólido con signo de cada cara vista desde ``p`` (fórmula de Van
    Oosterom y Strackee)."""
    a = V[F[:, 0]] - p
    b = V[F[:, 1]] - p
    c = V[F[:, 2]] - p
    la = np.linalg.norm(a, axis=-1)
    lb = np.linalg.norm(b, axis=-1)
    lc = np.linalg.norm(c, axis=-1)

    num = np.einsum("...i,...i->...", a, np.cross(b, c))
    den = (la * lb * lc
           + np.einsum("...i,...i->...", a, b) * lc
           + np.einsum("...i,...i->...", b, c) * la
           + np.einsum("...i,...i->...", c, a) * lb)
    return 2.0 * np.arctan2(num, den)


def winding_number(mesh: Mesh, points: np.ndarray) -> np.ndarray:
    P = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if P.shape[1] != 3:
        raise ValueError(f"points debe ser (k, 3), llegó {P.shape}")
    return _suma_directa(mesh.V, mesh.F, P)


def _suma_directa(V: np.ndarray, F: np.ndarray, P: np.ndarray) -> np.ndarray:
    out = np.zeros(P.shape[0])
    if len(F) == 0:
        return out
    paso = max(1, _CHUNK_ELEMS // len(F))
    for i in range(0, P.shape[0], paso):
        bloque = P[i:i + paso][:, None, :]
        out[i:i + paso] = solid_angles(V, F, bloque).sum(axis=1) * _INV_4PI
    return out


def _cierre(F: np.ndarray) -> np.ndarray:
    """Triángulos que cierran el pedazo de superficie ``F``.

    Cada arista que ``F`` recorre más veces en un sentido que en el otro (su
    borde) se une a un vértice fijo con un triángulo que la recorre al revés.
    Pedazo más cierre queda cerrado, que es lo único que hace falta.
    """
    d = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    signo = np.where(d[:, 0] < d[:, 1], 1, -1)
    claves, inv = np.unique(np.sort(d, axis=1), axis=0, return_inverse=True)
    cuenta = np.rint(np.bincount(inv.ravel(), weights=signo,
                                 minlength=len(claves))).astype(np.int64)
    borde = cuenta != 0
    i, j, c = claves[borde, 0], claves[borde, 1], cuenta[borde]

    k = int(F[0, 0])
    tri = np.where((c > 0)[:, None],
                   np.column_stack([j, i, np.full_like(i, k)]),
                   np.column_stack([i, j, np.full_like(i, k)]))
    tri = np.repeat(tri, np.abs(c), axis=0)
    # Los que tocan a k son degenerados.
    return tri[(tri[:, 0] != k) & (tri[:, 1] != k)]


@dataclass
class _Nodo:
    lo: np.ndarray           # caja de los vértices de sus caras
    hi: np.ndarray
    cierre: np.ndarray
    caras: np.ndarray | None  # solo en las hojas
    hijos: tuple[_Nodo, _Nodo] | None


class WindingHierarchy:
    """Evaluación jerárquica exacta (Algoritmos 1 y 2 del paper).

    Si un punto está fuera de la caja de un pedazo ``S``, ``w_S(p)`` es menos el
    número de vueltas de cualquier superficie que lo cierre, y el cierre tiene
    muchas menos caras que ``S``. El árbol parte la caja a la mitad por el lado
    más largo.
    """

    def __init__(self, mesh, leaf_size: int = 100) -> None:
        self.V = np.asarray(mesh.V, dtype=np.float64)
        self.leaf_size = leaf_size
        F = np.asarray(mesh.F, dtype=np.int64)
        self._eps = 1e-9 * float(np.linalg.norm(self.V.max(axis=0) - self.V.min(axis=0)))
        self.raiz = self._construir(F, *self._caja(F))

    def _caja(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        P = self.V[F.ravel()]
        return P.min(axis=0), P.max(axis=0)

    def _construir(self, F: np.ndarray, lo_part: np.ndarray, hi_part: np.ndarray) -> _Nodo:
        # La prueba de "lejos" usa la caja real de las caras, no la de la
        # partición, porque una cara puede asomar fuera de la caja de su hijo.
        lo, hi = self._caja(F)
        cierre = _cierre(F)
        if len(F) < self.leaf_size or len(cierre) >= len(F):
            return _Nodo(lo, hi, cierre, F, None)

        eje = int(np.argmax(hi_part - lo_part))
        corte = 0.5 * (lo_part[eje] + hi_part[eje])
        izq = self.V[F].mean(axis=1)[:, eje] < corte
        if izq.all() or not izq.any():
            return _Nodo(lo, hi, cierre, F, None)

        hi_izq, lo_der = hi_part.copy(), lo_part.copy()
        hi_izq[eje] = lo_der[eje] = corte
        hijos = (self._construir(F[izq], lo_part, hi_izq),
                 self._construir(F[~izq], lo_der, hi_part))
        return _Nodo(lo, hi, cierre, None, hijos)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        P = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if P.shape[1] != 3:
            raise ValueError(f"points debe ser (k, 3), llegó {P.shape}")
        out = np.zeros(P.shape[0])
        # Todos los puntos bajan juntos: en cada nodo los que quedan afuera de
        # la caja se resuelven con el cierre y el resto sigue a los hijos.
        pendientes = [(self.raiz, np.arange(P.shape[0]))]
        while pendientes:
            nodo, idx = pendientes.pop()
            Q = P[idx]
            if nodo.hijos is None:
                out[idx] += _suma_directa(self.V, nodo.caras, Q)
                continue
            afuera = np.any((Q < nodo.lo - self._eps) | (Q > nodo.hi + self._eps), axis=1)
            if afuera.any():
                out[idx[afuera]] -= _suma_directa(self.V, nodo.cierre, Q[afuera])
            adentro = idx[~afuera]
            if len(adentro):
                pendientes += [(h, adentro) for h in nodo.hijos]
        return out


def sample_grid(mesh: Mesh, resolution: int = 64, padding: float = 0.25,
                axis: int = 2, offset: float = 0.0):
    """Evalúa w en un corte plano de la caja envolvente.

    Devuelve ``(w, extent)``, listo para ``matplotlib.imshow``.
    """
    lo, hi = mesh.bounding_box()
    margen = padding * (hi - lo).max()
    lo, hi = lo - margen, hi + margen

    ejes = [i for i in range(3) if i != axis]
    u = np.linspace(lo[ejes[0]], hi[ejes[0]], resolution)
    v = np.linspace(lo[ejes[1]], hi[ejes[1]], resolution)
    U, W = np.meshgrid(u, v, indexing="xy")

    P = np.empty((U.size, 3))
    P[:, ejes[0]] = U.ravel()
    P[:, ejes[1]] = W.ravel()
    P[:, axis] = offset
    return winding_number(mesh, P).reshape(resolution, resolution), (u[0], u[-1], v[0], v[-1])
