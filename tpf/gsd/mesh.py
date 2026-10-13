"""Malla de triángulos con su conectividad.

Convención: la esquina ``i`` de una cara es opuesta a la arista
``F[f, i+1] -> F[f, i+2]``. Cada arista se guarda una vez como ``(min, max)`` y
``face_edge_sign[f, i]`` vale +1 si la cara la recorre en ese sentido y -1 si
no.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Para no dividir por cero al normalizar normales de triángulos degenerados.
_EPS_AREA = 1e-14


@dataclass
class Mesh:
    V: np.ndarray  # (n, 3) posiciones
    F: np.ndarray  # (m, 3) índices de vértices

    def __post_init__(self) -> None:
        self.V = np.ascontiguousarray(self.V, dtype=np.float64)
        self.F = np.ascontiguousarray(self.F, dtype=np.int32)
        if self.V.ndim != 2 or self.V.shape[1] != 3:
            raise ValueError(f"V debe ser (n, 3), llegó {self.V.shape}")
        if self.F.ndim != 2 or self.F.shape[1] != 3:
            raise ValueError(f"F debe ser (m, 3), llegó {self.F.shape}")
        if self.n_faces and (self.F.min() < 0 or self.F.max() >= self.n_vertices):
            raise ValueError("F tiene índices fuera de rango")
        self._build_edges()

    @property
    def n_vertices(self) -> int:
        return self.V.shape[0]

    @property
    def n_faces(self) -> int:
        return self.F.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edges.shape[0]

    def _build_edges(self) -> None:
        F = self.F
        dirigidas = np.stack(
            [np.stack([F[:, (i + 1) % 3], F[:, (i + 2) % 3]], axis=1) for i in range(3)],
            axis=1,
        ).reshape(-1, 2)

        self.edges, inv = np.unique(np.sort(dirigidas, axis=1), axis=0, return_inverse=True)
        self.face_edges = inv.reshape(-1, 3).astype(np.int32)
        self.face_edge_sign = np.where(dirigidas[:, 0] < dirigidas[:, 1], 1, -1
                                       ).reshape(-1, 3).astype(np.int8)
        # 1 = borde, 2 = interior, más de 2 = no manifold.
        self.edge_face_count = np.bincount(self.face_edges.ravel(),
                                           minlength=self.n_edges).astype(np.int32)

    @property
    def boundary_edges(self) -> np.ndarray:
        return np.flatnonzero(self.edge_face_count == 1)

    @property
    def is_edge_manifold(self) -> bool:
        return bool(np.all(self.edge_face_count <= 2))

    @property
    def is_closed(self) -> bool:
        return bool(np.all(self.edge_face_count == 2))

    @property
    def face_corners(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.V[self.F[:, 0]], self.V[self.F[:, 1]], self.V[self.F[:, 2]]

    @property
    def face_normals_raw(self) -> np.ndarray:
        """Normales sin normalizar (su norma es el doble del área)."""
        a, b, c = self.face_corners
        return np.cross(b - a, c - a)

    @property
    def face_areas(self) -> np.ndarray:
        return 0.5 * np.linalg.norm(self.face_normals_raw, axis=1)

    @property
    def face_normals(self) -> np.ndarray:
        n = self.face_normals_raw
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), _EPS_AREA)

    @property
    def total_area(self) -> float:
        return float(self.face_areas.sum())

    @property
    def corner_angles(self) -> np.ndarray:
        """(m, 3) ángulo interior de cada esquina."""
        p = self.V[self.F]
        angulos = np.empty((self.n_faces, 3))
        for i in range(3):
            u = p[:, (i + 1) % 3] - p[:, i]
            v = p[:, (i + 2) % 3] - p[:, i]
            # atan2 en vez de arccos: no pierde precisión cerca de 0 y de pi.
            angulos[:, i] = np.arctan2(np.linalg.norm(np.cross(u, v), axis=1),
                                       np.einsum("ij,ij->i", u, v))
        return angulos

    @property
    def edge_lengths(self) -> np.ndarray:
        e = self.edges
        return np.linalg.norm(self.V[e[:, 1]] - self.V[e[:, 0]], axis=1)

    @property
    def mean_edge_length(self) -> float:
        return float(self.edge_lengths.mean())

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.V.min(axis=0), self.V.max(axis=0)

    def transformed(self, R: np.ndarray | None = None, t: np.ndarray | None = None,
                    s: float = 1.0) -> Mesh:
        """Copia escalada, rotada y trasladada."""
        V = self.V * s
        if R is not None:
            V = V @ np.asarray(R, dtype=np.float64).T
        if t is not None:
            V = V + np.asarray(t, dtype=np.float64)
        return Mesh(V, self.F.copy())

    def with_flipped_orientation(self) -> Mesh:
        return Mesh(self.V.copy(), self.F[:, ::-1].copy())

    def without_faces(self, drop: np.ndarray) -> Mesh:
        keep = np.ones(self.n_faces, dtype=bool)
        keep[np.asarray(drop, dtype=np.int64)] = False
        return Mesh(self.V.copy(), self.F[keep].copy())

    def split_edges(self, partir: np.ndarray) -> Mesh:
        """Parte por el medio las aristas marcadas en ``partir`` (máscara sobre
        ``edges``) y retriangula cada cara según cuántas aristas se le partieron.

        Los vértices viejos conservan su índice y los puntos medios van al final.
        """
        partir = np.asarray(partir, dtype=bool)
        idx = np.flatnonzero(partir)
        medio = np.full(self.n_edges, -1, dtype=np.int64)
        medio[idx] = self.n_vertices + np.arange(len(idx))
        V = np.vstack([self.V, 0.5 * (self.V[self.edges[idx, 0]] + self.V[self.edges[idx, 1]])])

        F = self.F.astype(np.int64)
        m = medio[self.face_edges]  # m[:, k]: punto medio de la arista opuesta a k
        s = partir[self.face_edges]
        n = s.sum(axis=1)
        caras = [F[n == 0]]

        # Con una o dos aristas partidas se rota la cara para que la esquina 0
        # quede opuesta a la arista distinta (la partida o la que no).
        for cuantas, distinta in ((1, True), (2, False)):
            sel = n == cuantas
            k = np.argmax(s[sel] == distinta, axis=1)
            fila = np.arange(sel.sum())
            p0, p1, p2 = (F[sel][fila, (k + j) % 3] for j in range(3))
            m0, m1, m2 = (m[sel][fila, (k + j) % 3] for j in range(3))
            if cuantas == 1:
                nuevas = [(p0, p1, m0), (p0, m0, p2)]
            else:
                nuevas = [(p0, m2, m1), (m2, p1, p2), (m2, p2, m1)]
            caras += [np.column_stack(t) for t in nuevas]

        sel = n == 3
        a, b, c = F[sel].T
        ma, mb, mc = m[sel].T
        caras += [np.column_stack(t) for t in
                  ((a, mc, mb), (mc, b, ma), (mb, ma, c), (ma, mb, mc))]
        return Mesh(V, np.vstack(caras))


def load_obj(path: str | Path) -> Mesh:
    """Lee vértices y caras de un OBJ. Los polígonos se triangulan en abanico."""
    vertices: list[list[float]] = []
    caras: list[tuple[int, int, int]] = []

    with open(path, encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            if linea.startswith("v "):
                partes = linea.split()
                vertices.append([float(partes[1]), float(partes[2]), float(partes[3])])
            elif linea.startswith("f "):
                # Los tokens pueden ser "v", "v/vt", "v//vn" o "v/vt/vn".
                idx = [int(tok.split("/")[0]) for tok in linea.split()[1:]]
                # OBJ cuenta desde 1; los índices negativos cuentan desde el final.
                idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    caras.append((idx[0], idx[k], idx[k + 1]))

    if not caras:
        raise ValueError(f"{path} no tiene caras")
    return Mesh(np.array(vertices), np.array(caras))


def save_obj(mesh: Mesh, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for v in mesh.V:
            fh.write(f"v {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
        for f in mesh.F + 1:
            fh.write(f"f {f[0]} {f[1]} {f[2]}\n")
