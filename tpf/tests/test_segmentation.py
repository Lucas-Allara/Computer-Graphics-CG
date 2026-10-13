"""Segmentación por corte de grafo (Sección 5) y tetraedrizaciones."""

from itertools import product

import numpy as np
import pytest

from gsd.conforming import (
    conforming_tet_mesh,
    delaunay_tet_mesh,
    grid_tet_mesh,
    self_intersecting_faces,
)
from gsd.mesh import Mesh
from gsd.segmentation import (
    apply_hard_constraints,
    data_term,
    element_winding_number,
    estimate_sigma,
    extract_label_boundary,
    graphcut_segment,
    hard_constraints_from_surface,
    min_cut_labels,
    smoothness_weights,
    tet_adjacency,
)
from gsd.shapes import humanoid, icosphere, merge_meshes, random_holes, torus
from gsd.winding import winding_number
from gsd.tetmesh import tet_grid


# --------------------------------------------------------- adyacencia de tetraedros


def test_los_vecinos_comparten_una_cara_de_verdad():
    """Cada par comparte tres vértices y no falta ninguna cara interior."""
    mesh = tet_grid(n=4, size=2.0)
    tet_a, tet_b, area = tet_adjacency(mesh)

    assert len(tet_a) == int((mesh.face_tet_count == 2).sum())
    assert (area > 0).all()

    conjuntos = [set(map(int, mesh.T[t])) for t in range(mesh.n_tets)]
    for a, b in zip(tet_a, tet_b):
        assert len(conjuntos[a] & conjuntos[b]) == 3


def test_el_area_de_vecindad_es_la_de_la_cara_compartida():
    """``a_ij`` es el área de la cara compartida."""
    mesh = tet_grid(n=3, size=2.0)
    tet_a, tet_b, area = tet_adjacency(mesh)

    for k in range(0, len(tet_a), 37):
        comunes = set(map(int, mesh.T[tet_a[k]])) & set(map(int, mesh.T[tet_b[k]]))
        v = mesh.V[sorted(comunes)]
        area_directa = 0.5 * np.linalg.norm(
            np.cross(v[1] - v[0], v[2] - v[0])
        )
        assert abs(area[k] - area_directa) < 1e-9


# --------------------------------------------------------------- la energía


def test_termino_de_datos_ecuacion_9():
    w = np.array([-0.3, 0.0, 0.3, 0.5, 0.7, 1.0, 1.4])
    u_adentro, u_afuera = data_term(w)

    assert np.allclose(u_adentro, np.maximum(1 - w, 0))
    assert np.allclose(u_afuera, np.maximum(w, 0))
    assert u_adentro[w == 1.0] == 0.0
    assert u_afuera[w == 0.0] == 0.0


def test_termino_de_suavidad_decae_con_la_diferencia():
    """Separar dos vecinos cuesta menos cuanto más distinto es su w."""
    tet_a = np.array([0, 0, 0])
    tet_b = np.array([1, 1, 1])
    area = np.array([1.0, 1.0, 1.0])

    w = np.array([0.5, 0.5])
    w2 = np.array([0.55, 0.9])
    v1 = smoothness_weights(w, tet_a[:1], tet_b[:1], area[:1], sigma=0.2)
    v2 = smoothness_weights(w2, tet_a[:1], tet_b[:1], area[:1], sigma=0.2)
    assert v1[0] > v2[0]
    assert (smoothness_weights(np.array([0.5, 0.5]), np.array([0]), np.array([1]),
                               np.array([3.7]), sigma=0.2) <= 3.7 + 1e-12).all()


def test_estimate_sigma_es_positivo_y_no_colapsa_con_w_constante():
    tet_a, tet_b = np.array([0, 1]), np.array([1, 2])
    assert estimate_sigma(np.array([0.4, 0.4, 0.4]), tet_a, tet_b) > 0

    w = np.array([0.1, 0.5, 0.9])
    sigma = estimate_sigma(w, tet_a, tet_b, percentil=85.0)
    assert sigma == pytest.approx(
        np.percentile(np.abs(w[tet_a] - w[tet_b]), 85.0)
    )


def test_estimate_sigma_no_lo_domina_la_mayoria_lejana():
    """Con la mayoría de los pares iguales, la mediana sale diminuta y el
    percentil no."""
    rng = np.random.default_rng(5)
    lejos = rng.normal(0, 0.002, 700)
    cerca = rng.uniform(0.1, 0.6, 300)
    w = np.concatenate([np.zeros(1), lejos, cerca])
    tet_a = np.zeros(1000, dtype=np.int64)
    tet_b = np.arange(1, 1001, dtype=np.int64)

    mediana = np.median(np.abs(w[tet_a] - w[tet_b]))
    sigma = estimate_sigma(w, tet_a, tet_b)

    assert mediana < 0.01
    assert sigma > 0.05
    assert sigma > 50 * mediana


# ---------------------------------------------------- el corte mínimo en sí


def _energia(x, u_adentro, u_afuera, tet_a, tet_b, v):
    """Ecuación 8 evaluada en la etiqueta ``x``."""
    x = np.asarray(x, dtype=bool)
    costo = np.where(x, u_adentro, u_afuera).sum()
    costo += v[x[tet_a] != x[tet_b]].sum()
    return float(costo)


def test_el_corte_minimo_coincide_con_la_fuerza_bruta():
    """Contra la enumeración de las 2^m etiquetas."""
    rng = np.random.default_rng(0)
    for _ in range(30):
        m = int(rng.integers(3, 9))
        u_adentro = rng.uniform(0, 2, m)
        u_afuera = rng.uniform(0, 2, m)

        n_aristas = int(rng.integers(1, 2 * m))
        a = rng.integers(0, m, n_aristas)
        b = rng.integers(0, m, n_aristas)
        distintos = a != b
        a, b = a[distintos], b[distintos]
        v = rng.uniform(0, 3, len(a))

        etiqueta = min_cut_labels(u_adentro, u_afuera, a, b, v)
        encontrada = _energia(etiqueta, u_adentro, u_afuera, a, b, v)
        optima = min(
            _energia(x, u_adentro, u_afuera, a, b, v)
            for x in product([False, True], repeat=m)
        )

        assert encontrada == pytest.approx(optima, abs=1e-3)


def test_gamma_cero_es_exactamente_el_umbral():
    """Sin término de suavidad da el umbral ``w > 0.5`` (Ecuación 7)."""
    rng = np.random.default_rng(1)
    w = rng.uniform(-0.4, 1.4, size=60)
    u_adentro, u_afuera = data_term(w)

    a = rng.integers(0, 60, 150)
    b = rng.integers(0, 60, 150)
    v_nulo = np.zeros(150)

    etiqueta = min_cut_labels(u_adentro, u_afuera, a, b, v_nulo)
    assert np.array_equal(etiqueta, w > 0.5)


def test_restriccion_dura_fuerza_la_etiqueta_pase_lo_que_pase_con_el_dato():
    """Las restricciones se respetan y, entre las etiquetas que las cumplen,
    el corte da la de menor energía."""
    rng = np.random.default_rng(2)
    probados = 0
    while probados < 20:
        m = int(rng.integers(3, 8))
        u_adentro = rng.uniform(0, 2, m)
        u_afuera = rng.uniform(0, 2, m)
        n_aristas = int(rng.integers(1, 2 * m))
        a = rng.integers(0, m, n_aristas)
        b = rng.integers(0, m, n_aristas)
        distintos = a != b
        a, b = a[distintos], b[distintos]
        v = rng.uniform(0, 3, len(a))

        asignacion = rng.integers(-1, 2, m)  # -1 libre, 0 forzar afuera, 1 forzar adentro
        forzar_adentro = asignacion == 1
        forzar_afuera = asignacion == 0
        if not forzar_adentro.any() and not forzar_afuera.any():
            continue
        probados += 1

        ua2, uf2 = apply_hard_constraints(u_adentro, u_afuera, v,
                                          forzar_adentro, forzar_afuera)
        etiqueta = min_cut_labels(ua2, uf2, a, b, v)

        assert np.all(etiqueta[forzar_adentro])
        assert not np.any(etiqueta[forzar_afuera])

        compatibles = [
            x for x in product([False, True], repeat=m)
            if all(np.asarray(x)[forzar_adentro]) and not any(np.asarray(x)[forzar_afuera])
        ]
        optima = min(_energia(np.asarray(x), u_adentro, u_afuera, a, b, v)
                    for x in compatibles)
        encontrada = _energia(etiqueta, u_adentro, u_afuera, a, b, v)
        assert encontrada == pytest.approx(optima, abs=1e-3)


def test_restricciones_contradictorias_lanzan_error():
    u_adentro = np.array([1.0, 1.0])
    u_afuera = np.array([1.0, 1.0])
    v = np.array([0.5])
    with pytest.raises(ValueError):
        apply_hard_constraints(u_adentro, u_afuera, v,
                               forzar_adentro=np.array([True, False]),
                               forzar_afuera=np.array([True, False]))


def test_la_suavidad_corrige_ruido_incoherente_entre_vecinos():
    """Con w ruidoso alrededor de una transición, el corte mejora al umbral a
    medida que crece γ."""
    mesh = tet_grid(n=8, size=2.0)
    tet_a, tet_b, area = tet_adjacency(mesh)
    centros = mesh.tet_centroids

    w_real = 0.5 * (1 - np.tanh(centros[:, 0] / 0.35))
    rng = np.random.default_rng(3)
    w_ruidoso = np.clip(w_real + rng.normal(0, 0.22, size=len(w_real)), -0.4, 1.4)
    verdad = centros[:, 0] < 0

    u_adentro, u_afuera = data_term(w_ruidoso)
    sigma = estimate_sigma(w_ruidoso, tet_a, tet_b)

    acc_umbral = np.mean((w_ruidoso > 0.5) == verdad)
    assert acc_umbral < 0.96

    gammas = (0.0, 1.0, 5.0, 15.0, 40.0)
    precisiones = []
    for gamma in gammas:
        v = gamma * smoothness_weights(w_ruidoso, tet_a, tet_b, area, sigma)
        etiqueta = min_cut_labels(u_adentro, u_afuera, tet_a, tet_b, v)
        precisiones.append(np.mean(etiqueta == verdad))

    assert precisiones[0] == pytest.approx(acc_umbral, abs=1e-9)
    assert all(p2 >= p1 - 1e-9 for p1, p2 in zip(precisiones, precisiones[1:]))
    assert precisiones[-1] - precisiones[0] > 0.03


# ------------------------------------------------------- extracción de la superficie


def test_la_frontera_extraida_es_manifold_y_cierra():
    """Sobre una bola de tetraedros la frontera es cerrada y mira hacia afuera."""
    mesh = tet_grid(n=6, size=2.0)
    centros = mesh.tet_centroids
    adentro = np.linalg.norm(centros, axis=1) < 0.7

    V, F = extract_label_boundary(mesh, adentro)
    caras_usadas = np.unique(F)
    sub = Mesh(V[caras_usadas], np.searchsorted(caras_usadas, F))
    assert sub.is_closed
    assert len(F) > 0

    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    volumen = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    assert volumen > 0


# ------------------------------------------------- discretizaciones intercambiables


def _esferas_encastradas() -> Mesh:
    """Dos esferas que se interpenetran, desplazadas en una dirección genérica."""
    a = icosphere(2, radius=0.6)
    b = icosphere(2, radius=0.6)
    return merge_meshes([a, Mesh(b.V + np.array([0.55, 0.07, 0.03]), b.F)])


def _fuerza_bruta_cortes(surf: Mesh) -> np.ndarray:
    """Autointersecciones probando todas las parejas."""
    from gsd.conforming import _arista_atraviesa

    T = surf.V[surf.F]
    i, j = np.triu_indices(len(T), k=1)
    cortan = np.zeros(len(i), dtype=bool)
    for k in range(3):
        sig = (k + 1) % 3
        cortan |= _arista_atraviesa(T[i, k], T[i, sig], T[j, 0], T[j, 1], T[j, 2])
        cortan |= _arista_atraviesa(T[j, k], T[j, sig], T[i, 0], T[i, 1], T[i, 2])
    mascara = np.zeros(len(T), dtype=bool)
    mascara[i[cortan]] = True
    mascara[j[cortan]] = True
    return mascara


def test_una_malla_sana_no_tiene_autointersecciones():
    """Caras vecinas se tocan pero no se cortan."""
    for sana in (icosphere(3), torus(24, 12)):
        assert not self_intersecting_faces(sana).any()


def test_autointersecciones_casos_armados_a_mano():
    """Dos triángulos cruzados se cortan y uno lejano no. Un triángulo grande
    atravesado por uno chico también (caen en clases de radio distintas)."""
    V = np.array([
        [-1, -1, 0], [1, -1, 0], [0, 1, 0],          # horizontal
        [0, -0.2, -1], [0, 0.2, -1], [0, 0, 1],      # vertical, lo atraviesa
        [5, 5, 5], [6, 5, 5], [5, 6, 5],             # lejos
        [-30, -30, 2], [30, -30, 2], [0, 30, 2],     # enorme, en z = 2
        [0, 0, 1.9], [0.1, 0, 2.1], [0, 0.1, 2.1],   # chico, cruza z = 2
    ], dtype=float)
    F = np.arange(15).reshape(5, 3)
    mascara = self_intersecting_faces(Mesh(V, F))
    assert mascara.tolist() == [True, True, False, True, True]


def test_la_busqueda_de_pares_no_se_pierde_ningun_corte():
    surf = _esferas_encastradas()
    esperado = _fuerza_bruta_cortes(surf)
    obtenido = self_intersecting_faces(surf)
    assert esperado.sum() > 0
    assert np.array_equal(obtenido, esperado)


def test_el_refinamiento_rojo_verde_es_conforme_y_no_mueve_la_superficie():
    """La esfera sigue cerrada, con la misma área y volumen."""
    from gsd.conforming import _refinar_rojo_verde

    esfera = icosphere(2)
    rng = np.random.default_rng(3)
    marcadas = rng.random(esfera.n_faces) < 0.1
    V, F = _refinar_rojo_verde(esfera.V, esfera.F, marcadas)
    fina = Mesh(V, F)

    assert fina.is_closed
    assert fina.total_area == pytest.approx(esfera.total_area, rel=1e-12)

    def volumen(m):
        a, b, c = m.V[m.F[:, 0]], m.V[m.F[:, 1]], m.V[m.F[:, 2]]
        return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0

    assert volumen(fina) == pytest.approx(volumen(esfera), rel=1e-12)
    assert fina.n_faces > esfera.n_faces + 3 * marcadas.sum()


def test_restricciones_automaticas_no_inventan_nada_sobre_una_rejilla():
    """En una rejilla ninguna cara de entrada es cara de la malla."""
    rota = random_holes(icosphere(2), target_fraction=0.15, n_holes=4, seed=7)
    forzar_adentro, forzar_afuera = hard_constraints_from_surface(
        rota, grid_tet_mesh(rota, n=9))
    assert not forzar_adentro.any()
    assert not forzar_afuera.any()


def _segmentar_esfera_rota(mesh_de):
    """Segmenta la esfera con agujeros sobre ``mesh_de(rota)`` y controla
    que la malla cubra la entrada y la salida sea cerrada."""
    rota = random_holes(icosphere(2), target_fraction=0.15, n_holes=4, seed=7)
    mesh = mesh_de(rota)

    lo, hi = rota.bounding_box()
    assert np.all(mesh.V.min(axis=0) <= lo) and np.all(mesh.V.max(axis=0) >= hi)

    resultado = graphcut_segment(rota, mesh=mesh, gamma=3.0)
    assert Mesh(resultado["V"], resultado["F"]).is_closed
    return mesh, resultado


def test_la_segmentacion_corre_sobre_delaunay_sin_restricciones():
    """Delaunay no agrega puntos: la salida cae exactamente sobre la esfera."""
    _, resultado = _segmentar_esfera_rota(delaunay_tet_mesh)
    desvio = np.abs(np.linalg.norm(resultado["V"], axis=1) - 1.0)
    assert desvio.max() < 1e-12


def test_la_segmentacion_corre_sobre_una_rejilla():
    """Sobre una rejilla la salida se aparta de la esfera del orden de una
    celda (sobre los agujeros, más)."""
    n = 14
    mesh, resultado = _segmentar_esfera_rota(lambda s: grid_tet_mesh(s, n=n))
    celda = float((mesh.V.max(axis=0) - mesh.V.min(axis=0)).max()) / n
    desvio = np.abs(np.linalg.norm(resultado["V"], axis=1) - 1.0)
    assert desvio.mean() < 0.6 * celda


# ---------------------------------------------------------------- pipeline completo


tetgen = pytest.importorskip("tetgen", reason="el pipeline completo necesita TetGen")


def test_conforming_tet_mesh_preserva_cada_triangulo_de_entrada():
    """Cada triángulo de entrada es una cara de la malla, con los mismos
    vértices."""
    esfera = icosphere(3, radius=0.5)
    mesh, ids = conforming_tet_mesh(esfera)

    d = np.linalg.norm(mesh.V[ids] - esfera.V, axis=1)
    assert d.max() < 1e-12

    caras_malla = {tuple(c) for c in np.sort(mesh.faces, axis=1)}
    faltantes = sum(
        tuple(sorted(int(ids[v]) for v in tri)) not in caras_malla
        for tri in np.asarray(esfera.F)
    )
    assert faltantes == 0


def test_graphcut_segment_reconstruye_una_esfera_con_agujeros():
    sana = icosphere(2)
    rota = random_holes(sana, target_fraction=0.15, n_holes=4, seed=7)

    resultado = graphcut_segment(rota, gamma=3.0, margin=0.22, max_far_edge=0.35)
    assert len(resultado["F"]) > 0

    area = Mesh(resultado["V"], resultado["F"]).total_area
    assert 0.85 < area / sana.total_area < 1.30


def test_hard_constraints_from_surface_no_se_pisan_y_marcan_lo_intacto():
    """Las restricciones no se pisan, y coinciden con lo que ya dice w."""
    sana = icosphere(2)
    rota = random_holes(sana, target_fraction=0.15, n_holes=4, seed=7)
    mesh, _ = conforming_tet_mesh(rota, margin=0.22, max_far_edge=0.35)

    forzar_adentro, forzar_afuera = hard_constraints_from_surface(rota, mesh)
    assert not np.any(forzar_adentro & forzar_afuera)
    assert forzar_adentro.sum() > 0
    assert forzar_afuera.sum() > 0

    w = element_winding_number(rota, mesh)
    assert w[forzar_adentro].mean() > 0.9
    assert w[forzar_afuera].mean() < 0.1


def test_restricciones_duras_no_empeoran_el_area_reconstruida():
    sana = icosphere(2)
    rota = random_holes(sana, target_fraction=0.15, n_holes=4, seed=7)
    mesh, _ = conforming_tet_mesh(rota, margin=0.22, max_far_edge=0.35)

    libre = graphcut_segment(rota, mesh=mesh, gamma=3.0)
    forzado = graphcut_segment(rota, mesh=mesh, gamma=3.0, restricciones_duras=True)

    area_libre = Mesh(libre["V"], libre["F"]).total_area / sana.total_area
    area_forzado = Mesh(forzado["V"], forzado["F"]).total_area / sana.total_area
    assert abs(area_forzado - 1.0) < abs(area_libre - 1.0) + 0.05


def test_extraer_dos_veces_con_la_misma_malla_es_determinista():
    sana = icosphere(1)
    mesh, _ = conforming_tet_mesh(sana, margin=0.25, max_far_edge=0.4)

    r1 = graphcut_segment(sana, mesh=mesh, gamma=2.0)
    r2 = graphcut_segment(sana, mesh=mesh, gamma=2.0)
    assert np.array_equal(r1["adentro"], r2["adentro"])


@pytest.mark.parametrize("rondas", [0, 2])
def test_conforming_tet_mesh_admite_autointersecciones(rondas):
    """Con caras que se cortan, las que no se cortan siguen siendo caras de la
    malla. Al subdividir, las vecinas de las que se cortan se bisectan y de
    esas no se exige nada."""
    surf = _esferas_encastradas()
    cortan = self_intersecting_faces(surf)
    mesh, ids = conforming_tet_mesh(surf, refine_intersections=rondas)

    assert np.abs(mesh.V[ids] - surf.V).max() == 0.0
    caras_malla = {tuple(c) for c in np.sort(mesh.faces, axis=1)}
    presentes = np.array([tuple(sorted(int(ids[v]) for v in tri)) in caras_malla
                          for tri in surf.F])
    if rondas == 0:
        assert presentes[~cortan].all()
    else:
        tocadas = np.isin(surf.F, surf.F[cortan]).any(axis=1)
        assert presentes[~tocadas].all()


def _fraccion_de_frontera_correcta(surf: Mesh, resultado: dict) -> float:
    """Fracción del área de salida con ``w > ½`` detrás y ``w < ½`` delante.
    Una membrana en el aire o una cáscara enterrada no lo cumplen."""
    salida = Mesh(resultado["V"], resultado["F"])
    centros = salida.V[salida.F].mean(axis=1)
    paso = 0.05 * surf.mean_edge_length * salida.face_normals
    delante = winding_number(surf, centros + paso)
    detras = winding_number(surf, centros - paso)
    bien = (detras > 0.5) & (delante < 0.5)
    return float(salida.face_areas[bien].sum() / salida.total_area)


def test_subdividir_las_intersecciones_quita_las_membranas():
    """En el humanoide, sin subdividir quedan membranas entre los miembros."""
    hum = humanoid(subdivisions=1)
    lo, hi = hum.bounding_box()
    lejos = float(np.linalg.norm(hi - lo)) / 15

    fracciones = []
    for rondas in (0, 2):
        mesh, _ = conforming_tet_mesh(hum, max_far_edge=lejos,
                                      refine_intersections=rondas)
        resultado = graphcut_segment(hum, mesh=mesh, gamma=3.0, quadrature=1)
        fracciones.append(_fraccion_de_frontera_correcta(hum, resultado))

    assert fracciones[1] > 0.95
    assert fracciones[1] > fracciones[0] + 0.1


def test_piezas_encastradas_dan_la_union():
    """La salida es la frontera de la unión, y las restricciones automáticas
    no la cambian."""
    surf = _esferas_encastradas()
    mesh, _ = conforming_tet_mesh(surf, max_far_edge=0.05)

    libre = graphcut_segment(surf, mesh=mesh, gamma=3.0)
    # Sin bordes, aunque sobre la curva de intersección puede haber aristas
    # con cuatro caras.
    F = libre["F"]
    aristas = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, valencia = np.unique(aristas, axis=0, return_counts=True)
    assert np.all(valencia % 2 == 0)
    assert _fraccion_de_frontera_correcta(surf, libre) > 0.98

    r, d = 0.6, float(np.linalg.norm([0.55, 0.07, 0.03]))
    casquete = 2 * np.pi * r * (r - d / 2)
    union = 2 * (4 * np.pi * r ** 2) - 2 * casquete
    assert Mesh(libre["V"], libre["F"]).total_area == pytest.approx(union, rel=0.03)

    forzado = graphcut_segment(surf, mesh=mesh, gamma=3.0, restricciones_duras=True)
    assert np.array_equal(forzado["adentro"], libre["adentro"])


def test_las_heuristicas_de_la_seccion_5_2():
    """Sobre la CDT de la cápsula convexa, el corte deja aristas no variedad y
    caras de entrada afuera. Las heurísticas arreglan las dos cosas."""
    from gsd.restricciones import caras_de_entrada

    rota = random_holes(icosphere(3), target_fraction=0.15, n_holes=4, seed=7)
    usados, F = np.unique(rota.F, return_inverse=True)
    rota = Mesh(rota.V[usados], F.reshape(-1, 3))
    mesh, _ = conforming_tet_mesh(rota)
    entrada = caras_de_entrada(rota, mesh)

    libre = graphcut_segment(rota, mesh=mesh, gamma=3.0, quadrature=1)
    con = graphcut_segment(rota, mesh=mesh, gamma=3.0, quadrature=1, heuristicas=True)
    n_libre = np.bincount(mesh.tet_faces[libre["adentro"]].ravel(), minlength=mesh.n_faces)
    n_con = np.bincount(mesh.tet_faces[con["adentro"]].ravel(), minlength=mesh.n_faces)

    assert not Mesh(libre["V"], libre["F"]).is_closed
    assert (n_libre[entrada] == 0).any()
    assert Mesh(con["V"], con["F"]).is_closed
    assert (n_con[entrada] > 0).all()


def test_las_heuristicas_no_tocan_una_salida_valida():
    sana = icosphere(2)
    mesh, _ = conforming_tet_mesh(sana, max_far_edge=0.35)
    libre = graphcut_segment(sana, mesh=mesh, gamma=3.0)
    con = graphcut_segment(sana, mesh=mesh, gamma=3.0, heuristicas=True)
    assert np.array_equal(libre["adentro"], con["adentro"])
