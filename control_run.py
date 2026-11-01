"""One random-perturbation control replicate in a fresh process."""
import sys, os
import pipeline_schubert as ps
from config_v2 import cargar_comunidad
rep = int(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
dest = "resultados/control_rep%02d.csv" % rep
if not os.path.exists(dest):
    m, mapa = cargar_comunidad()
    c = ps.control_escenarios(m, mapa, B=1, n=n, eps=0.02, cache_dir="cache", inicio=rep)
    c.to_csv(dest, index=False)
    print("DONE control rep", rep)
