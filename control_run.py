"""One random-perturbation control replicate in a fresh process."""
import sys, os, pickle, pandas as pd, pipeline_schubert as ps
rep = int(sys.argv[1]); n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
dest = "resultados/control_rep%02d.csv" % rep
if os.path.exists(dest):
    sys.exit(0)
m = ps.cargar_modelo("community12_fixed.xml", solver="hybrid"); ps.aplicar_medio(m)
mapa = pickle.load(open("mapa_todas.pkl", "rb"))
ps.control_escenarios(m, mapa, B=1, n=n, cache_dir="cache", inicio=rep).to_csv(dest, index=False)
print("DONE control rep", rep)
