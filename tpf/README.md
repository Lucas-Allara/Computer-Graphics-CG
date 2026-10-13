# TPF — Segmentación robusta con números de vueltas generalizados

Implementación de *Robust Inside-Outside Segmentation using Generalized Winding
Numbers* (Jacobson, Kavan y Sorkine-Hornung, SIGGRAPH 2013), con dos agregados:
suavizado de las tapas de los agujeros y completación por simetría.

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Tests

```bash
.venv/bin/python -m pytest
```

## Uso

```python
from gsd import load_obj, reconstruct_with_symmetry

entrada = load_obj("assets/spot/spot_triangulated.obj")
salida = reconstruct_with_symmetry(entrada)
V, F = salida["simetria"] or salida["suavizado"]   # o salida["metodo"]
```

## Demos

Cada demo escribe su figura en `figuras/`.

```bash
.venv/bin/python demos/demo_winding.py
.venv/bin/python demos/demo_pieza_faltante.py
.venv/bin/python demos/demo_segmentacion.py
.venv/bin/python demos/demo_segmentacion_spot.py
.venv/bin/python demos/demo_niveles_de_dano.py             # unos 10 minutos
.venv/bin/python demos/demo_niveles_de_dano.py --figuras   # solo redibuja lo guardado
```

## Póster

Necesita los datos que guarda `demo_niveles_de_dano.py` y `lualatex` con los paquetes
`fira`, `firamath` y `qrcode`.

```bash
.venv/bin/python poster/figuras_poster.py
poster/compilar.sh
```
