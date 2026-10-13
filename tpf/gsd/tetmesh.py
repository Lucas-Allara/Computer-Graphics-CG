"""Malla de tetraedros. La cara ``i`` de un tetraedro es la opuesta a su
vértice ``i``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS_VOL = 1e-14


@dataclass
class TetMesh:
    V: np.ndarray  # (n, 3) posiciones
    T: np.ndarray  # (m, 4) índices de vértices

    def __post_init__(self) -> None:
        self.V = np.ascontiguousarray(self.V, dtype=np.float64)
        self.T = np.ascontiguousarray(self.T, dtype=np.int32)
        if self.V.ndim != 2 or self.V.shape[1] != 3:
            raise ValueError(f"V debe ser (n, 3), llegó {self.V.shape}")
        if self.T.ndim != 2 or self.T.shape[1] != 4:
            raise ValueError(f"T debe ser (m, 4), llegó {self.T.shape}")
        if self.n_tets and (self.T.min() < 0 or self.T.max() >= self.n_vertices):
            raise ValueError("T tiene índices fuera de rango")
        self._orient()
        self._build_faces()

    @property
    def n_vertices(self) -> int:
        return self.V.shape[0]

    @property
    def n_tets(self) -> int:
        return self.T.shape[0]

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]

    def _signed_volumes(self) -> np.ndarray:
        p = self.V[self.T]
        return np.linalg.det(p[:, 1:] - p[:, :1]) / 6.0

    def _orient(self) -> None:
        """Da vuelta los tetraedros con volumen negativo."""
        negativos = self._signed_volumes() < 0
        if negativos.any():
            self.T[negativos] = self.T[negativos][:, [0, 2, 1, 3]]
        if np.abs(self._signed_volumes()).min() < _EPS_VOL:
            raise ValueError("hay tetraedros degenerados (volumen nulo)")

    def _build_faces(self) -> None:
        T = self.T
        caras = np.stack([np.stack([T[:, (i + 1) % 4], T[:, (i + 2) % 4], T[:, (i + 3) % 4]], axis=1)
                          for i in range(4)], axis=1).reshape(-1, 3)
        self.faces, inv = np.unique(np.sort(caras, axis=1), axis=0, return_inverse=True)
        self.tet_faces = inv.reshape(-1, 4).astype(np.int32)
        # 1 = cara del borde del dominio, 2 = interior.
        self.face_tet_count = np.bincount(self.tet_faces.ravel(),
                                          minlength=self.n_faces).astype(np.int32)

    @property
    def face_areas(self) -> np.ndarray:
        f = self.V[self.faces]
        return 0.5 * np.linalg.norm(np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]), axis=1)

    @property
    def tet_centroids(self) -> np.ndarray:
        return self.V[self.T].mean(axis=1)


def drop_degenerate_tets(V: np.ndarray, T: np.ndarray,
                         tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Saca los tetraedros de volumen casi nulo (TetGen y Delaunay a veces los
    dejan)."""
    V = np.asarray(V, dtype=np.float64)
    T = np.asarray(T, dtype=np.int32)
    p = V[T]
    volumen = np.abs(np.linalg.det(p[:, 1:] - p[:, :1])) / 6.0
    return V, T[volumen > tol]


def tet_grid(n: int = 8, size: float = 2.0, center: np.ndarray | None = None) -> TetMesh:
    """Cubo de lado ``size`` partido en ``n³`` cubitos, y cada cubito en seis
    tetraedros (subdivisión de Kuhn)."""
    ejes = np.linspace(-size / 2, size / 2, n + 1)
    X, Y, Z = np.meshgrid(ejes, ejes, ejes, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    # Cada permutación de los ejes es un camino de (0,0,0) a (1,1,1) por aristas
    # del cubo; sus cuatro vértices forman uno de los seis tetraedros.
    caminos = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        c = np.zeros(3, dtype=np.int64)
        camino = [c.copy()]
        for eje in perm:
            c[eje] += 1
            camino.append(c.copy())
        caminos.append(camino)
    caminos = np.array(caminos)  # (6, 4, 3)

    base = np.stack(np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij"),
                    axis=-1).reshape(-1, 1, 1, 3)
    esquinas = base + caminos  # (n³, 6, 4, 3)
    T = ((esquinas[..., 0] * (n + 1) + esquinas[..., 1]) * (n + 1) + esquinas[..., 2]).reshape(-1, 4)

    c = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
    return TetMesh(V + c, T)
