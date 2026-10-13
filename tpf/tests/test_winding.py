"""Número de vueltas generalizado, directo y jerárquico."""

import numpy as np
import pytest

from gsd import (
    Mesh,
    cube,
    icosphere,
    punch_hole,
    tetrahedron,
    torus,
    winding_number,
)

CLOSED_SHAPES = [
    ("tetraedro", tetrahedron(), np.zeros(3)),
    ("cubo", cube(), np.zeros(3)),
    ("icosfera", icosphere(2), np.zeros(3)),
    ("toro", torus(32, 16), np.array([1.0, 0.0, 0.0])),  # sobre el tubo, no el hueco
]


# --------------------------------------------------------------- malla cerrada


@pytest.mark.parametrize("name,mesh,inside", CLOSED_SHAPES)
def test_vale_uno_adentro(name, mesh, inside):
    w = winding_number(mesh, inside[None, :])[0]
    assert np.isclose(w, 1.0, atol=1e-9), f"{name}: adentro dio {w}"


@pytest.mark.parametrize("name,mesh,inside", CLOSED_SHAPES)
def test_vale_cero_afuera(name, mesh, inside):
    lejos = np.array([[10.0, 7.0, -13.0]])
    w = winding_number(mesh, lejos)[0]
    assert np.isclose(w, 0.0, atol=1e-9), f"{name}: afuera dio {w}"


def test_el_hueco_del_toro_esta_afuera():
    """El origen está en el agujero del toro, fuera del sólido."""
    w = winding_number(torus(32, 16), np.zeros((1, 3)))[0]
    assert np.isclose(w, 0.0, atol=1e-9)


def test_salto_unitario_al_cruzar_la_superficie():
    """Al atravesar la superficie, w salta una unidad."""
    mesh = icosphere(3)
    n = np.array([0.0, 0.0, 1.0])
    adentro = winding_number(mesh, (n * 0.995)[None, :])[0]
    afuera = winding_number(mesh, (n * 1.005)[None, :])[0]
    assert np.isclose(adentro - afuera, 1.0, atol=0.02)


def test_transicion_abrupta_en_malla_sana():
    """Sobre una malla cerrada la transición ocurre en una sola muestra."""
    mesh = icosphere(3)
    t = np.linspace(0.0, 2.0, 200)
    P = np.stack([np.zeros_like(t), np.zeros_like(t), t], axis=1)
    w = winding_number(mesh, P)
    assert np.max(np.abs(np.diff(w))) > 0.8


# --------------------------------------------------------------- orientación


def test_dar_vuelta_la_malla_invierte_el_signo():
    mesh = icosphere(2)
    P = np.array([[0.0, 0.0, 0.0], [0.3, 0.2, -0.1], [5.0, 5.0, 5.0]])
    assert np.allclose(
        winding_number(mesh, P), -winding_number(mesh.with_flipped_orientation(), P)
    )


# --------------------------------------------------------------- invariancias


def test_invariante_ante_movimiento_rigido():
    mesh = icosphere(2)
    P = np.array([[0.0, 0.0, 0.0], [0.5, 0.1, 0.2], [1.4, 0.0, 0.0]])

    theta = 0.9
    R = np.array(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]]
    )
    t = np.array([-2.0, 4.0, 1.5])

    assert np.allclose(winding_number(mesh, P), winding_number(mesh.transformed(R=R, t=t), P @ R.T + t))


def test_invariante_ante_escala():
    mesh = icosphere(2)
    P = np.array([[0.0, 0.0, 0.0], [0.5, 0.1, 0.2], [3.0, 0.0, 0.0]])
    assert np.allclose(winding_number(mesh, P), winding_number(mesh.transformed(s=7.0), P * 7.0))


def test_invariante_ante_refinar_la_malla():
    P = np.array([[0.0, 0.0, 0.0], [0.4, 0.3, 0.2], [2.0, 0.0, 0.0]])
    a = winding_number(icosphere(2), P)
    b = winding_number(icosphere(4), P)
    assert np.allclose(a, b, atol=1e-6)


# ------------------------------------------------------------- geometría rota


def test_agujerear_baja_el_valor_de_adentro():
    """Sin un casquete, w en el centro baja en la fracción de área que falta."""
    mesh = icosphere(4)
    holed = punch_hole(mesh, np.array([0.0, 0.0, 1.0]), 0.5)

    w = winding_number(holed, np.zeros((1, 3)))[0]
    assert 0.85 < w < 0.99, f"valor en el centro: {w}"

    faltante = 1.0 - holed.total_area / mesh.total_area
    assert np.isclose(1.0 - w, faltante, atol=0.02)


def test_agujero_mas_grande_baja_mas_el_valor():
    mesh = icosphere(4)
    valores = [
        winding_number(punch_hole(mesh, np.array([0.0, 0.0, 1.0]), r), np.zeros((1, 3)))[0]
        for r in (0.3, 0.6, 0.9)
    ]
    assert valores[0] > valores[1] > valores[2]


def test_la_transicion_se_suaviza_cerca_del_hueco():
    """Sobre un segmento que pasa por el agujero, w cambia de a poco en vez de
    saltar."""
    mesh = icosphere(3)
    holed = punch_hole(mesh, np.array([0.0, 0.0, 1.0]), 0.6)

    t = np.linspace(0.5, 1.5, 300)
    P = np.stack([np.zeros_like(t), np.zeros_like(t), t], axis=1)

    salto_sano = np.max(np.abs(np.diff(winding_number(mesh, P))))
    salto_roto = np.max(np.abs(np.diff(winding_number(holed, P))))

    assert salto_sano > 0.8, "la malla cerrada debería saltar una unidad"
    assert salto_roto < 0.1, "la malla rota no debería saltar"


def test_lejos_del_hueco_sigue_clasificando_bien():
    """Del lado opuesto al agujero, w sigue cerca de 1."""
    holed = punch_hole(icosphere(4), np.array([0.0, 0.0, 1.0]), 0.5)
    w = winding_number(holed, np.array([[0.0, 0.0, -0.9]]))[0]
    assert w > 0.9


def test_decae_a_cero_lejos():
    holed = punch_hole(icosphere(3), np.array([0.0, 0.0, 1.0]), 0.6)
    lejanos = np.array([[50.0, 0.0, 0.0], [0.0, 80.0, 0.0], [0.0, 0.0, -60.0]])
    assert np.allclose(winding_number(holed, lejanos), 0.0, atol=1e-3)


# ------------------------------------------------------------------ mecánica


def test_consulta_por_bloques_da_lo_mismo():
    from gsd import winding

    mesh = icosphere(3)
    rng = np.random.default_rng(0)
    P = rng.normal(size=(500, 3))

    esperado = winding_number(mesh, P)
    original = winding._CHUNK_ELEMS
    try:
        winding._CHUNK_ELEMS = 1000
        assert np.allclose(winding_number(mesh, P), esperado)
    finally:
        winding._CHUNK_ELEMS = original


def test_forma_de_la_salida():
    mesh = tetrahedron()
    assert winding_number(mesh, np.zeros((1, 3))).shape == (1,)
    assert winding_number(mesh, np.zeros((17, 3))).shape == (17,)
    with pytest.raises(ValueError):
        winding_number(mesh, np.zeros((4, 2)))


# --------------------------------------------------------- evaluación jerárquica


def _entradas_de_todo_tipo():
    from gsd.shapes import humanoid, merge_meshes, mobius_strip, random_holes

    esfera = icosphere(3)
    return {
        "cerrada": esfera,
        "con agujeros": random_holes(icosphere(3), 0.2, 4, seed=2),
        "piezas que se cortan": humanoid(subdivisions=1),
        "no orientable": mobius_strip(),
        "duplicada": merge_meshes([esfera, esfera]),
        "con caras dadas vuelta": Mesh(esfera.V, np.where(
            (np.arange(esfera.n_faces) % 7 == 0)[:, None], esfera.F[:, ::-1], esfera.F)),
    }


@pytest.mark.parametrize("nombre", list(_entradas_de_todo_tipo()))
def test_la_jerarquia_da_lo_mismo_que_la_suma_directa(nombre):
    """Coincide salvo redondeo en cualquier entrada, lejos y pegado a la
    superficie."""
    from gsd.winding import WindingHierarchy

    malla = _entradas_de_todo_tipo()[nombre]
    rng = np.random.default_rng(0)
    lo, hi = malla.bounding_box()
    lejos = lo + (hi - lo) * (rng.random((1500, 3)) * 1.6 - 0.3)
    centros = malla.V[malla.F].mean(axis=1)
    pegados = centros + 1e-3 * rng.standard_normal(centros.shape)
    P = np.vstack([lejos, pegados])

    jerarquia = WindingHierarchy(malla, leaf_size=20)
    assert np.abs(jerarquia(P) - winding_number(malla, P)).max() < 1e-10


def test_el_cierre_de_un_pedazo_lo_cierra():
    """Pedazo más cierre recorre cada arista lo mismo en los dos sentidos."""
    from gsd.winding import _cierre

    pedazo = icosphere(2).F[:150]
    tapa = _cierre(pedazo)
    todas = np.vstack([pedazo, tapa])
    d = np.concatenate([todas[:, [0, 1]], todas[:, [1, 2]], todas[:, [2, 0]]])
    signo = np.where(d[:, 0] < d[:, 1], 1, -1)
    _, inv = np.unique(np.sort(d, axis=1), axis=0, return_inverse=True)
    assert np.all(np.bincount(inv.ravel(), weights=signo) == 0)
    assert len(tapa) < len(pedazo)
