# -*- coding: utf-8 -*-
"""
Dependencies of Salinibacter ruber and Haloquadratum walsbyi on the community
(part 3 of ecologia.py; uses resultados/eco_red_pfba.csv).

For each candidate metabolite: maximal growth of the taxon when its NET uptake
of that metabolite is forbidden (community growth free, and >= 90% optimum).
For Haloquadratum also: what it takes from the community when it grows at its
community maximum (pFBA with Growth_Hqr fixed).

Outputs: resultados/eco_dependencias.csv, resultados/eco_hqr_importa.csv
Usage (Jupyter):  %run ecologia_dep.py      (about 10 min; needs eco_red_pfba.csv and eco_hqr_importa.csv)
"""
import pickle
import time
import numpy as np
import pandas as pd
import cobra
import pipeline_schubert as ps
import curar_modelo as cm

from config_v2 import cargar_comunidad
EPS = 0.02

m, mapa = cargar_comunidad(acoplar=False)
import sys
m.solver.configuration.lp_method = sys.argv[1] if len(sys.argv) > 1 else "interior point"   # simplex stalls here
m.solver.configuration.timeout = 900
print("lp methods", m.solver.configuration.lp_methods, flush=True)
cm.acoplar(m, mapa=mapa)
sufs = cm.sufijos(m)
gro = {s.split("_")[0]: m.reactions.get_by_id("Growth_" + s) for s in sufs}
opt = m.slim_optimize()
W = mapa.W.tocsr()
medio = set(k + "_e" for k in ps.M9_GLICEROL) | {"glyc_e"}
print("community optimum %.4f" % opt, flush=True)


def restriccion_sin_captacion(especie, met):
    """Constraint: net exchange of `especie` on `met` >= 0 (no net uptake)."""
    i = mapa.especies.index(especie)
    j = mapa.met_e.index(met)
    fila = W.getrow(i * mapa.m + j)
    c = m.problem.Constraint(0, lb=0.0, name="block_%s_%s" % (especie, met))
    m.add_cons_vars(c)
    coefs = {}
    for k, a in zip(fila.indices, fila.data):
        r = m.reactions.get_by_id(mapa.cols[k])
        coefs[r.forward_variable] = coefs.get(r.forward_variable, 0) + a
        coefs[r.reverse_variable] = coefs.get(r.reverse_variable, 0) - a
    c.set_linear_coefficients(coefs)
    return c


red = pd.read_csv("resultados/eco_red_pfba.csv")

imp = pd.read_csv("resultados/eco_hqr_importa.csv")   # written by the first run
# --- blocking tests -----------------------------------------------------
cands = {
    "Salinibacter": set(red[(red.receptor == "Salinibacter") & ~red.en_medio].metabolito)
                    | {"lys__L_e", "orn_e", "arg__L_e"},
    "Haloquadratum": set(imp[~imp.en_medio].metabolito),
}
dep = []
for sp, mets in cands.items():
    for met in sorted(x for x in mets if x in mapa.met_e):
        res = {"especie": sp, "metabolito": met}
        for etiqueta, lb in (("libre", 0.0), ("al_90", 0.9 * opt)):
            t0 = time.time()
            with m:
                restriccion_sin_captacion(sp, met)
                m.reactions.community_growth.lower_bound = lb
                m.objective = gro[sp]
                res["max_" + etiqueta] = m.slim_optimize(error_value=float("nan"))
                res["t_" + etiqueta] = time.time() - t0
        dep.append(res)
        print("%-13s block %-14s max %.3f (free)  %.3f (90%%)  [%.0fs]"
              % (sp, met, res["max_libre"], res["max_al_90"], time.time() - t0), flush=True)
        pd.DataFrame(dep).to_csv("resultados/eco_dependencias.csv", index=False)
# all non-medium uptake blocked at once (Salinibacter): is it auxotrophic?
with m:
    for met in cands["Salinibacter"]:
        restriccion_sin_captacion("Salinibacter", met)
    m.objective = gro["Salinibacter"]
    print("Salinibacter, all community-derived uptake blocked: %.3f" % m.slim_optimize(error_value=0.0))
print("ECOLOGY FINISHED")
