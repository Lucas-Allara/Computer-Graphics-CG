"""Conectividad y geometría básica de Mesh."""

import numpy as np
import pytest

from gsd import Mesh, cube, icosphere, load_obj, punch_hole, save_obj, tetrahedron, torus

CLOSED_SHAPES = [
    ("tetraedro", tetrahedron()),
    ("cubo", cube()),
    ("icosfera", icosphere(2)),
    ("toro", torus(16, 8)),
]


@pytest.mark.parametrize("name,mesh", CLOSED_SHAPES)
def test_euler_y_cierre(name, mesh):
    """Toda arista tocada por exactamente dos caras, y Euler-Poincaré se cumple."""
    assert mesh.is_closed, f"{name} debería ser cerrada"
    assert mesh.is_edge_manifold

    # V - E + F = 2 - 2g. El toro tiene género 1, el resto género 0.
    chi = mesh.n_vertices - mesh.n_edges + mesh.n_faces
    expected = 0 if name == "toro" else 2
    assert chi == expected, f"{name}: característica de Euler {chi}, se esperaba {expected}"


@pytest.mark.parametrize("name,mesh", CLOSED_SHAPES)
def test_cada_arista_recorrida_en_ambos_sentidos(name, mesh):
    """En una malla cerrada y orientada, las dos caras de una arista la recorren
    en sentidos opuestos."""
    total = np.zeros(mesh.n_edges, dtype=np.int64)
    np.add.at(total, mesh.face_edges.ravel(), mesh.face_edge_sign.ravel().astype(np.int64))
    assert np.all(total == 0), f"{name}: hay aristas recorridas dos veces en el mismo sentido"


@pytest.mark.parametrize("name,mesh", CLOSED_SHAPES)
def test_angulos_de_cada_cara_suman_pi(name, mesh):
    ang = mesh.corner_angles
    assert np.allclose(ang.sum(axis=1), np.pi)
    assert np.all(ang > 0)


def test_area_de_la_esfera_converge():
    """Al refinar, el área de la icosfera tiende a la de la esfera unidad."""
    errors = [
        abs(icosphere(k).total_area - 4 * np.pi) / (4 * np.pi) for k in (1, 2, 3)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.01


def test_normales_de_la_esfera_apuntan_hacia_afuera():
    mesh = icosphere(2)
    centroids = mesh.V[mesh.F].mean(axis=1)
    outward = np.einsum("ij,ij->i", mesh.face_normals, centroids)
    assert np.all(outward > 0), "hay caras con la normal hacia adentro"


def test_agujerear_deja_borde():
    mesh = icosphere(2)
    holed = punch_hole(mesh, np.array([0.0, 0.0, 1.0]), 0.5)
    assert holed.n_faces < mesh.n_faces
    assert not holed.is_closed
    assert len(holed.boundary_edges) > 0


def test_invariancia_ante_similitudes():
    """Los ángulos no cambian al rotar, trasladar y escalar; el área escala con
    el cuadrado."""
    mesh = icosphere(2)
    theta = 0.7
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    moved = mesh.transformed(R=R, t=np.array([3.0, -1.0, 2.0]), s=2.0)

    assert np.allclose(mesh.corner_angles, moved.corner_angles)
    assert np.isclose(moved.total_area, mesh.total_area * 4.0)


def test_dar_vuelta_la_orientacion_invierte_las_normales():
    mesh = icosphere(1)
    flipped = mesh.with_flipped_orientation()
    assert np.allclose(mesh.face_normals, -flipped.face_normals)
    assert np.allclose(mesh.face_areas, flipped.face_areas)


def test_ida_y_vuelta_por_obj(tmp_path):
    mesh = icosphere(1)
    path = tmp_path / "esfera.obj"
    save_obj(mesh, path)
    reloaded = load_obj(path)

    assert np.allclose(mesh.V, reloaded.V)
    assert np.array_equal(mesh.F, reloaded.F)
    assert reloaded.n_edges == mesh.n_edges


def test_obj_con_indices_negativos_y_poligonos(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "f -4 -3 -2 -1\n",  # un cuadrilátero con índices negativos
        encoding="utf-8",
    )
    mesh = load_obj(path)
    assert mesh.n_faces == 2  # triangulado en abanico
    assert np.isclose(mesh.total_area, 1.0)


def test_malla_no_manifold_se_detecta():
    """Tres triángulos que comparten una arista."""
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    F = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]], dtype=np.int32)
    mesh = Mesh(V, F)
    assert not mesh.is_edge_manifold
    assert mesh.edge_face_count.max() == 3


def test_indices_fuera_de_rango_fallan():
    with pytest.raises(ValueError):
        Mesh(np.zeros((3, 3)), np.array([[0, 1, 5]]))


@pytest.mark.parametrize("fraccion", [0.05, 0.3, 1.0])
def test_partir_aristas_no_deja_vertices_colgados(fraccion):
    """Con caras de una, dos y tres aristas partidas, la malla sigue cerrada,
    con la misma área y volumen, y los vértices viejos no se mueven."""
    esfera = icosphere(2)
    partir = np.random.default_rng(1).random(esfera.n_edges) < fraccion
    fina = esfera.split_edges(partir)

    assert fina.is_closed
    assert fina.n_vertices == esfera.n_vertices + partir.sum()
    assert np.array_equal(fina.V[: esfera.n_vertices], esfera.V)
    assert fina.total_area == pytest.approx(esfera.total_area, rel=1e-12)

    def volumen(m):
        a, b, c = m.V[m.F[:, 0]], m.V[m.F[:, 1]], m.V[m.F[:, 2]]
        return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0

    assert volumen(fina) == pytest.approx(volumen(esfera), rel=1e-12)
