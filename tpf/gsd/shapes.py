"""Mallas de prueba generadas por código y formas de romperlas."""

from __future__ import annotations

import numpy as np

from .mesh import Mesh

_PHI = (1.0 + np.sqrt(5.0)) / 2.0

# Icosaedro con las caras orientadas hacia afuera.
_ICO_V = np.array([
    [-1, _PHI, 0], [1, _PHI, 0], [-1, -_PHI, 0], [1, -_PHI, 0],
    [0, -1, _PHI], [0, 1, _PHI], [0, -1, -_PHI], [0, 1, -_PHI],
    [_PHI, 0, -1], [_PHI, 0, 1], [-_PHI, 0, -1], [-_PHI, 0, 1],
], dtype=np.float64)

_ICO_F = np.array([
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
], dtype=np.int32)


def icosphere(subdivisions: int = 2, radius: float = 1.0) -> Mesh:
    """Esfera por subdivisión de un icosaedro."""
    V = _ICO_V / np.linalg.norm(_ICO_V[0])
    F = _ICO_F.copy()

    for _ in range(subdivisions):
        puntos = list(V)
        medio: dict[tuple[int, int], int] = {}

        def mid(a: int, b: int) -> int:
            clave = (a, b) if a < b else (b, a)
            if clave not in medio:
                m = (puntos[a] + puntos[b]) / 2.0
                puntos.append(m / np.linalg.norm(m))
                medio[clave] = len(puntos) - 1
            return medio[clave]

        nuevas = []
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nuevas += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V = np.array(puntos)
        F = np.array(nuevas, dtype=np.int32)

    return Mesh(V * radius, F)


def cube(size: float = 1.0) -> Mesh:
    h = size / 2.0
    V = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ])
    F = np.array([
        [0, 3, 2], [0, 2, 1], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ])
    return Mesh(V, F)


def tetrahedron(scale: float = 1.0) -> Mesh:
    V = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3.0)
    F = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    return Mesh(V * scale, F)


def torus(n_major: int = 32, n_minor: int = 16, R: float = 1.0, r: float = 0.35) -> Mesh:
    u = np.arange(n_major) * (2 * np.pi / n_major)
    v = np.arange(n_minor) * (2 * np.pi / n_minor)
    U, W = np.meshgrid(u, v, indexing="ij")
    V = np.stack([((R + r * np.cos(W)) * np.cos(U)).ravel(),
                  ((R + r * np.cos(W)) * np.sin(U)).ravel(),
                  (r * np.sin(W)).ravel()], axis=1)

    def vid(i: int, j: int) -> int:
        return (i % n_major) * n_minor + (j % n_minor)

    F = []
    for i in range(n_major):
        for j in range(n_minor):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            F += [[a, b, c], [a, c, d]]
    return Mesh(V, np.array(F))


def mobius_strip(n_around: int = 64, n_across: int = 8, radius: float = 1.0,
                 width: float = 0.5) -> Mesh:
    """Banda de Möbius: superficie no orientable y con borde."""
    u = np.arange(n_around) * (2 * np.pi / n_around)
    v = np.linspace(-width / 2, width / 2, n_across + 1)
    U, W = np.meshgrid(u, v, indexing="ij")
    V = np.stack([((radius + W * np.cos(U / 2)) * np.cos(U)).ravel(),
                  ((radius + W * np.cos(U / 2)) * np.sin(U)).ravel(),
                  (W * np.sin(U / 2)).ravel()], axis=1)

    def vid(i: int, j: int) -> int:
        return (i % n_around) * (n_across + 1) + j

    F = []
    for i in range(n_around):
        ultima = i == n_around - 1
        for j in range(n_across):
            a, d = vid(i, j), vid(i, j + 1)
            # La última columna se pega dada vuelta: ahí se pierde la orientabilidad.
            jb, jc = (n_across - j, n_across - j - 1) if ultima else (j, j + 1)
            F += [[a, vid(i + 1, jb), vid(i + 1, jc)], [a, vid(i + 1, jc), d]]
    return Mesh(V, np.array(F))


def ellipsoid(center=(0.0, 0.0, 0.0), radii=(1.0, 1.0, 1.0), subdivisions: int = 3) -> Mesh:
    base = icosphere(subdivisions)
    return Mesh(base.V * np.asarray(radii, float) + np.asarray(center, float), base.F.copy())


def capsule(a, b, radius: float, subdivisions: int = 3) -> Mesh:
    """Cápsula entre los puntos ``a`` y ``b``: una icosfera partida por el ecuador
    y con las dos mitades llevadas a los extremos."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    largo = float(np.linalg.norm(b - a))
    if largo < 1e-12:
        return ellipsoid(a, (radius,) * 3, subdivisions)
    base = icosphere(subdivisions)
    hacia_b = (base.V @ ((b - a) / largo)) > 0
    return Mesh(base.V * radius + np.where(hacia_b[:, None], b, a), base.F.copy())


def merge_meshes(meshes) -> Mesh:
    """Junta varias mallas en una sola, sin unirlas: las piezas quedan
    superpuestas como estén."""
    V, F, desplazamiento = [], [], 0
    for m in meshes:
        V.append(m.V)
        F.append(m.F + desplazamiento)
        desplazamiento += m.n_vertices
    return Mesh(np.vstack(V), np.vstack(F))


def humanoid(scale: float = 1.0, subdivisions: int = 3) -> Mesh:
    """Muñeco armado con piezas que se interpenetran (cabeza, torso, brazos,
    manos, piernas y pies)."""
    s = scale
    piezas = [
        ellipsoid((0, 0, 0.50 * s), (0.15 * s, 0.14 * s, 0.16 * s), subdivisions),
        ellipsoid((0, 0, 0.17 * s), (0.20 * s, 0.12 * s, 0.24 * s), subdivisions),
    ]
    for lado in (-1, 1):
        piezas += [
            capsule((lado * 0.17 * s, 0, 0.33 * s), (lado * 0.40 * s, 0, 0.02 * s),
                    0.075 * s, subdivisions),
            ellipsoid((lado * 0.43 * s, 0, -0.03 * s), (0.075 * s,) * 3, subdivisions),
            capsule((lado * 0.10 * s, 0, -0.03 * s), (lado * 0.13 * s, 0, -0.48 * s),
                    0.090 * s, subdivisions),
            ellipsoid((lado * 0.13 * s, 0.05 * s, -0.54 * s),
                      (0.085 * s, 0.14 * s, 0.055 * s), subdivisions),
        ]
    return merge_meshes(piezas)


def punch_hole(mesh: Mesh, center: np.ndarray, radius: float) -> Mesh:
    """Saca las caras cuyo baricentro cae a menos de ``radius`` de ``center``."""
    d = np.linalg.norm(mesh.V[mesh.F].mean(axis=1) - np.asarray(center, float), axis=1)
    return mesh.without_faces(np.flatnonzero(d < radius))


def random_holes(mesh: Mesh, target_fraction: float, n_holes: int = 3,
                 seed: int = 0) -> Mesh:
    """Hace ``n_holes`` agujeros en lugares al azar hasta sacar más o menos la
    fracción de área pedida.

    Ojo: las caras se borran pero los vértices quedan, sueltos adentro de los
    agujeros.
    """
    if target_fraction <= 0:
        return Mesh(mesh.V.copy(), mesh.F.copy())

    rng = np.random.default_rng(seed)
    area_inicial = mesh.total_area
    actual = mesh
    escala = np.sqrt(target_fraction * area_inicial / (n_holes * np.pi))

    for k in range(n_holes):
        if actual.n_faces < 8:
            break
        centro = actual.V[actual.F].mean(axis=1)[rng.integers(actual.n_faces)]
        objetivo = area_inicial * target_fraction * (k + 1) / n_holes
        radio, tentativa = escala, actual
        for _ in range(24):
            tentativa = punch_hole(actual, centro, radio)
            if area_inicial - tentativa.total_area >= objetivo or tentativa.n_faces < 8:
                break
            radio *= 1.15
        actual = tentativa if tentativa.n_faces >= 8 else actual

    return actual


def connected_components(mesh: Mesh) -> int:
    """Cantidad de componentes conexas (por vértices compartidos)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components as componentes

    e = mesh.edges
    g = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(mesh.n_vertices,) * 2)
    return int(componentes(g, directed=False)[0])
