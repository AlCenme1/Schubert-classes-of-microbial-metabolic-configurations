"""One sampling batch (40 new states) in a fresh process; resumes from cache/."""
import sys
import pipeline_schubert as ps
from config_v2 import cargar_comunidad
esc = sys.argv[1]
m, mapa = cargar_comunidad()
ps.muestrear(m, mapa, esc, n=150, cache_dir="cache", guardar_cada=10, max_nuevos=40)
