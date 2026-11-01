import sys, glob, time, cobra, pandas as pd, numpy as np, pipeline_schubert as ps
from cobra.flux_analysis import pfba
import curar_modelo as cm
ENERGIA = cm.ENERGIA
filas = []
for f in sorted(glob.glob("indiv/*.xml")):
    t = time.time()
    m = cobra.io.read_sbml_model(f); m.solver = "glpk"
    nom = f.split("/")[-1][:-4]
    bio = [r for r in m.reactions if r.id in ("Growth", "R_Growth")] or [r for r in m.reactions if "growth" in r.id.lower()]
    bio = bio[0]
    met_rx = [r for r in m.reactions if not r.boundary and r is not bio]
    rev = sum(1 for r in met_rx if r.lower_bound < 0 < r.upper_bound)
    sincarga = sum(1 for x in m.metabolites if x.charge is None)
    imb = sum(1 for r in met_rx if abs(r.check_mass_balance().get("charge", 0) or 0) > 1e-6)
    ex = [r for r in m.reactions if r.boundary and r.id.startswith("EX_")]
    # growth M9[glyc] (same medium definition as the pipeline) and without carbon
    with m:
        ps.aplicar_medio(m, ps.M9_GLICEROL, verbose=False); m.objective = bio
        g = m.slim_optimize(error_value=0.0)
    with m:
        ps.aplicar_medio(m, {k: v for k, v in ps.M9_GLICEROL.items() if k != "glyc"}, verbose=False); m.objective = bio
        g0 = m.slim_optimize(error_value=0.0)
    # EGC test, closed system
    egc = []
    with m:
        for r in m.reactions:
            if r.boundary: r.bounds = (0, 0)
        for k, (izq, der) in ENERGIA.items():
            mets = {}
            ok = True
            for mid, c in list(izq.items()) + list(der.items()):
                if mid not in m.metabolites: ok = False; break
                mets[m.metabolites.get_by_id(mid)] = c
            if not ok: continue
            d = cobra.Reaction("DISS_" + k, lower_bound=0, upper_bound=1000); d.add_metabolites(mets)
            with m:
                m.add_reactions([d]); m.objective = d
                if m.slim_optimize(error_value=0.0) > 1e-6: egc.append(k)
    # glycerol route and gene support
    gl = []
    for rid in ("GLYCt", "GLYCtex", "GLYCtpp", "GLYK", "G3PD2", "G3PD5", "G3PD7", "GLYCDx", "GLYCDy", "DHAK", "GLYCabc"):
        if rid in m.reactions:
            r = m.reactions.get_by_id(rid)
            gl.append("%s%s" % (rid, "" if r.gene_reaction_rule else "*"))
    filas.append(dict(modelo=nom, rxns=len(m.reactions), metab_rxn=len(met_rx), mets=len(m.metabolites),
                      genes=len(m.genes), pct_rev=round(100*rev/len(met_rx), 1), sin_carga=sincarga,
                      desbal_carga=imb, n_EX=len(ex), biomasa=bio.id, crec_M9glyc=round(g, 3),
                      crec_sin_C=round(g0, 4), EGC=",".join(egc) or "none", glicerol=" ".join(gl),
                      seg=round(time.time()-t)))
    print(filas[-1], flush=True)
pd.DataFrame(filas).to_csv("inspeccion_individuales.csv", index=False)
