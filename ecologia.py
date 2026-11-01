# -*- coding: utf-8 -*-
"""
Ecological reading of the curated 17-taxon community on M9 + glycerol
(coupled, Heinken et al. 2013).

1. Growth of each taxon alone vs its maximal growth inside the community
   (others unconstrained). Taxa that grow only inside the community are
   obligate cross-feeders on this medium.
2. Parsimonious cross-feeding network at 90% of the community optimum
   (pFBA): donor -> receiver -> metabolite.
Dependency tests (blocking uptake) are in ecologia_dep.py.

Outputs: resultados/eco_crecimiento.csv, eco_red_pfba.csv
Usage (Jupyter):  %run ecologia.py     (about 1.5 GB RAM, ~15 min)
"""
import pickle
import numpy as np
import pandas as pd
import cobra
import pipeline_schubert as ps
import curar_modelo as cm

from config_v2 import cargar_comunidad
EPS = 0.02

m, mapa = cargar_comunidad(acoplar=False)
cm.acoplar(m, mapa=mapa)
sufs = cm.sufijos(m)
tok = {s: s.split("_")[0] for s in sufs}
opt = m.slim_optimize()
print("community optimum %.4f" % opt)

# --- 1. alone vs inside the community --------------------------------
filas = []
for s in sufs:
    g = m.reactions.get_by_id("Growth_" + s)
    with m:
        m.objective = g
        libre = m.slim_optimize(error_value=0.0)
    with m:
        m.reactions.community_growth.lower_bound = 0.9 * opt
        m.objective = g
        a90 = m.slim_optimize(error_value=0.0)
    filas.append({"especie": tok[s], "max_en_comunidad": libre, "max_al_90": a90})
    print("%-18s community max %.3f   at 90%% opt %.3f" % (tok[s], libre, a90), flush=True)
cre = pd.DataFrame(filas)

# --- 2. parsimonious network ------------------------------------------
sol = cobra.flux_analysis.pfba(m, fraction_of_optimum=0.9)
v = np.array([sol.fluxes.get(c, 0.0) for c in mapa.cols])
X = mapa.flujos_netos(v)
cre["mu_pfba"] = [sol.fluxes["Growth_" + s] for s in sufs]
cre.to_csv("resultados/eco_crecimiento.csv", index=False)
medio = set(k + "_e" for k in ps.M9_GLICEROL) | {"glyc_e"}
red = []
for j in np.where((X > EPS).any(0) & (X < -EPS).any(0))[0]:
    don = np.where(X[:, j] > EPS)[0]
    rec = np.where(X[:, j] < -EPS)[0]
    tot = X[don, j].sum()
    for r in rec:
        for d in don:
            red.append({"donante": mapa.especies[d], "receptor": mapa.especies[r],
                        "metabolito": mapa.met_e[j], "en_medio": mapa.met_e[j] in medio,
                        "flujo": -X[r, j] * X[d, j] / tot})
red = pd.DataFrame(red).sort_values("flujo", ascending=False)
red.to_csv("resultados/eco_red_pfba.csv", index=False)
print("\nparsimonious cross-feeding (non-medium metabolites):")
print(red[~red.en_medio].round(3).to_string(index=False))

print("ECOLOGY (parts 1-2) FINISHED; dependency tests: ecologia_dep.py")
