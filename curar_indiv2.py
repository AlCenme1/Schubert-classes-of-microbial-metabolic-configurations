from egc import egc_ciclos
import glob, pickle, cobra, pandas as pd, pipeline_schubert as ps, curar_modelo as cm
from reglas import reversibles_con_oxidante, fijar_direccion

res = {}
import sys
ARCH = sys.argv[1:] or sorted(glob.glob("indiv/*.xml"))
for f in ARCH:
    nom = f.split("/")[-1][:-4]
    m = cobra.io.read_sbml_model(f); m.solver = "glpk"; m.objective = "Growth"
    with m:
        ps.aplicar_medio(m, ps.M9_GLICEROL, verbose=False); g_antes = m.slim_optimize(error_value=0)
    cambios = []
    for r, s in reversibles_con_oxidante(m):
        antes = r.bounds; fijar_direccion(r, s)
        cambios.append(dict(reaccion=r.id, regla="oxidase irreversible", antes=antes, despues=r.bounds))
    for r in m.reactions:
        b = r.id.split("_")[0]
        if b in cm.FIJAS and r.lower_bound < 0 < r.upper_bound:
            antes = r.bounds; fijar_direccion(r, cm.FIJAS[b])
            cambios.append(dict(reaccion=r.id, regla="EGC fix", antes=antes, despues=r.bounds))
    with m:
        ps.aplicar_medio(m, ps.M9_GLICEROL, verbose=False); g = m.slim_optimize(error_value=0)
    with m:
        ps.aplicar_medio(m, {k: v for k, v in ps.M9_GLICEROL.items() if k != "glyc"}, verbose=False); g0 = m.slim_optimize(error_value=0)
    ciclos = egc_ciclos(m)
    res[nom] = dict(cambios=cambios, crec_antes=g_antes, crec=g, crec_sinC=g0, egc=ciclos)
    print(f"\n=== {nom}: oxidase/EGC fixes {len(cambios)} {[c['reaccion'] for c in cambios]}\n    growth M9glyc {g_antes:.3f} -> {g:.3f}; no carbon {g0:.4f}; EGC {list(ciclos)}", flush=True)
    vistos = set()
    for k, cyc in ciclos.items():
        key = tuple(sorted(cyc))
        if key in vistos: continue
        vistos.add(key)
        print(f"   {k}: {cyc}")
        for rid in cyc:
            r = m.reactions.get_by_id(rid)
            if r.lower_bound < 0 < r.upper_bound:
                print(f"       rev {rid}: {r.reaction[:120]}")
    pickle.dump(m, open(f"ind2_{nom}.pkl", "wb"))
pickle.dump(res, open("curar_indiv2_res.pkl", "wb"))
