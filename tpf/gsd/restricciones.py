"""Restricciones opcionales de la Sección 5.2 del paper.

No se pueden imponer dentro del corte de grafo, así que el paper las aplica con
dos heurísticas sobre la etiqueta que da el corte:

- Cada cara de entrada tiene que ser cara de algún tetraedro de adentro. Si no
  lo es, se pasa adentro su tetraedro vecino de mayor w.
- La superficie de salida tiene que ser una variedad. En cada arista o vértice
  donde no lo es, se pasan adentro los tetraedros que lo tocan, de mayor a menor
  w, hasta que lo sea.

Después de cada paso se mejora la energía de la Ecuación 8 dando vuelta vecinos
de lo que cambió, sin romper restricciones que ya se cumplían.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .mesh import Mesh
from .tetmesh import TetMesh


def caras_de_entrada(surf: Mesh, mesh: TetMesh) -> np.ndarray:
    """Índices en ``mesh.faces`` de los triángulos de ``surf`` que son caras de
    la tetraedrización (con vértices exactamente iguales)."""
    from scipy.spatial import cKDTree

    d, ids = cKDTree(mesh.V).query(np.asarray(surf.V))
    diagonal = float(np.linalg.norm(mesh.V.max(axis=0) - mesh.V.min(axis=0)))
    F = np.asarray(surf.F)
    F = F[(d <= 1e-12 * diagonal)[F].all(axis=1)]
    indice = {tuple(c): k for k, c in enumerate(np.sort(mesh.faces, axis=1).tolist())}
    caras = (indice.get(tuple(sorted(int(ids[v]) for v in t))) for t in F)
    return np.array([c for c in caras if c is not None], dtype=np.int64)


def _csr(claves: np.ndarray, valores: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    orden = np.argsort(claves, kind="stable")
    return np.searchsorted(claves[orden], np.arange(n + 1)), valores[orden]


class _Etiquetado:
    """La etiqueta de cada tetraedro y lo necesario para cambiarla de a uno."""

    def __init__(self, mesh: TetMesh, w, adentro, u_adentro, u_afuera, tet_a, tet_b, v_pesos,
                 entrada: np.ndarray) -> None:
        m, nf = mesh.n_tets, mesh.n_faces
        self.w = np.asarray(w)
        self.x = np.asarray(adentro, dtype=bool).copy()
        self.costo = (np.asarray(u_afuera), np.asarray(u_adentro))
        self.T = mesh.T
        self.tet_caras = mesh.tet_faces.astype(np.int64)
        self.variedad = False

        tf = self.tet_caras.ravel()
        tet = np.repeat(np.arange(m), 4)
        orden = np.argsort(tf, kind="stable")
        tf_o, tet_o = tf[orden], tet[orden]
        primero = np.ones(len(tf_o), dtype=bool)
        primero[1:] = tf_o[1:] != tf_o[:-1]
        self.cara_tets = np.full((nf, 2), -1, dtype=np.int64)
        self.cara_tets[tf_o[primero], 0] = tet_o[primero]
        self.cara_tets[tf_o[~primero], 1] = tet_o[~primero]
        # Tetraedros de adentro de cada cara: la cara es de la salida si hay uno solo.
        self.n_adentro = np.bincount(tf[self.x[tet]], minlength=nf)

        self.es_entrada = np.zeros(nf, dtype=bool)
        self.es_entrada[entrada] = True

        caras = mesh.faces.astype(np.int64)
        aristas = np.sort(np.concatenate([caras[:, [0, 1]], caras[:, [1, 2]], caras[:, [2, 0]]]), axis=1)
        self.aristas, inv = np.unique(aristas, axis=0, return_inverse=True)
        self.cara_aristas = inv.reshape(3, nf).T
        ne = len(self.aristas)
        self.arista_caras = _csr(self.cara_aristas.ravel(), np.repeat(np.arange(nf), 3), ne)
        self.vertice_caras = _csr(caras.ravel(), np.repeat(np.arange(nf), 3), mesh.n_vertices)
        self.vertice_tets = _csr(self.T.ravel().astype(np.int64), tet, mesh.n_vertices)
        self.tet_aristas = np.sort(self.cara_aristas[self.tet_caras].reshape(m, 12), axis=1)[:, ::2]
        self.vecinos = _csr(np.concatenate([tet_a, tet_b]),
                            np.stack([np.concatenate([tet_b, tet_a]),
                                      np.arange(2 * len(tet_a)) % len(tet_a)], axis=1), m)
        self.v_pesos = np.asarray(v_pesos)

    @staticmethod
    def _de(csr, k):
        inicio, valores = csr
        return valores[inicio[k]:inicio[k + 1]]

    def dar_vuelta(self, i: int) -> None:
        self.x[i] = not self.x[i]
        self.n_adentro[self.tet_caras[i]] += 1 if self.x[i] else -1

    def arista_ok(self, e: int) -> bool:
        return int((self.n_adentro[self._de(self.arista_caras, e)] == 1).sum()) in (0, 2)

    def vertice_ok(self, v: int) -> bool:
        caras = self._de(self.vertice_caras, v)
        caras = caras[self.n_adentro[caras] == 1]
        if len(caras) == 0:
            return True
        # Las caras de salida alrededor de v tienen que formar un solo abanico,
        # unidas por las aristas que contienen a v.
        padre = list(range(len(caras)))

        def raiz(a):
            while padre[a] != a:
                padre[a] = padre[padre[a]]
                a = padre[a]
            return a

        primera = {}
        for k, c in enumerate(caras):
            for e in self.cara_aristas[c]:
                if v in self.aristas[e]:
                    if e in primera:
                        padre[raiz(k)] = raiz(primera[e])
                    else:
                        primera[e] = k
        return len({raiz(k) for k in range(len(caras))}) == 1

    def delta(self, i: int) -> float:
        actual = bool(self.x[i])
        vec = self._de(self.vecinos, i)
        j, peso = vec[:, 0], self.v_pesos[vec[:, 1]]
        xj = self.x[j]
        return float(self.costo[not actual][i] - self.costo[actual][i]
                     + (peso * ((xj == actual).astype(float) - (xj != actual))).sum())

    def intentar(self, i: int) -> bool:
        """Da vuelta ``i`` si no rompe ninguna restricción que se cumplía."""
        if self.x[i]:
            caras = self.tet_caras[i]
            if np.any(self.es_entrada[caras] & (self.n_adentro[caras] == 1)):
                return False
        if not self.variedad:
            self.dar_vuelta(i)
            return True
        aristas, vertices = self.tet_aristas[i], self.T[i]
        antes = [self.arista_ok(e) for e in aristas] + [self.vertice_ok(v) for v in vertices]
        self.dar_vuelta(i)
        despues = [self.arista_ok(e) for e in aristas] + [self.vertice_ok(v) for v in vertices]
        if any(a and not d for a, d in zip(antes, despues)):
            self.dar_vuelta(i)
            return False
        return True

    def mejorar(self, cambiados) -> None:
        """Baja la energía dando vuelta vecinos de los tetraedros cambiados."""
        cola = deque(int(j) for i in cambiados for j in self._de(self.vecinos, i)[:, 0])
        while cola:
            j = cola.popleft()
            if self.delta(j) < -1e-12 and self.intentar(j):
                cola.extend(int(k) for k in self._de(self.vecinos, j)[:, 0])

    def pasar_adentro(self, tets, cumple) -> None:
        """Pasa adentro los tetraedros de afuera de ``tets``, de mayor a menor w,
        hasta que ``cumple()``."""
        tets = tets[~self.x[tets]]
        cambiados = []
        for i in tets[np.argsort(-self.w[tets], kind="stable")]:
            if cumple():
                break
            self.dar_vuelta(int(i))
            cambiados.append(int(i))
        self.mejorar(cambiados)


def enforce_optional_constraints(mesh: TetMesh, entrada: np.ndarray, w, adentro,
                                 u_adentro, u_afuera, tet_a, tet_b, v_pesos,
                                 max_rondas: int = 50) -> np.ndarray:
    """Aplica las dos heurísticas de la Sección 5.2 a la etiqueta ``adentro``.

    ``entrada`` son los índices de las caras de ``mesh`` que vienen de la entrada
    (``caras_de_entrada``). Devuelve la etiqueta nueva.
    """
    e = _Etiquetado(mesh, w, adentro, u_adentro, u_afuera, tet_a, tet_b, v_pesos, entrada)

    for f in entrada:
        if e.n_adentro[f] == 0:
            tets = e.cara_tets[f][e.cara_tets[f] >= 0]
            i = int(tets[np.argmax(e.w[tets])])
            e.dar_vuelta(i)
            e.mejorar([i])

    e.variedad = True
    for _ in range(max_rondas):
        salida = e.n_adentro == 1
        cuenta = np.bincount(e.cara_aristas[salida].ravel(), minlength=len(e.aristas))
        malas = np.flatnonzero((cuenta != 0) & (cuenta != 2))
        for a in malas:
            if not e.arista_ok(a):
                tets = np.unique(e.cara_tets[e._de(e.arista_caras, a)])
                e.pasar_adentro(tets[tets >= 0], lambda a=a: e.arista_ok(a))
        malos = [v for v in np.unique(mesh.faces[e.n_adentro == 1]) if not e.vertice_ok(v)]
        for v in malos:
            if not e.vertice_ok(v):
                e.pasar_adentro(e._de(e.vertice_tets, v), lambda v=v: e.vertice_ok(v))
        if len(malas) == 0 and len(malos) == 0:
            break
    return e.x
