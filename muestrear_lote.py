"""One batch (<= 40 new states) of random-objective FBA, resumed from cache."""
import sys, pickle, pipeline_schubert as ps
esc = sys.argv[1]
modelo = sys.argv[2] if len(sys.argv) > 2 else "community12_fixed.xml"
interfaz = sys.argv[3] if len(sys.argv) > 3 else "todas"
n = int(sys.argv[4]) if len(sys.argv) > 4 else 150
import numpy as np, os
ruta = "cache/flujos_%s_n%d.npz" % (esc, n)
while not (os.path.exists(ruta) and len(np.load(ruta)["F"]) >= n):
    import subprocess
    r = subprocess.run([sys.executable, "-c", f"""
import pickle, pipeline_schubert as ps
m = ps.cargar_modelo({modelo!r}, solver="hybrid"); ps.aplicar_medio(m)
mapa = pickle.load(open("mapa_{interfaz}.pkl", "rb"))
ps.muestrear(m, mapa, {esc!r}, n={n}, cache_dir="cache", guardar_cada=10, max_nuevos=40)
"""])
    if r.returncode != 0:
        sys.exit(r.returncode)
print("DONE", esc)
