# -*- coding: utf-8 -*-
"""
Cross-feeding network across the sampled states (both scenarios).

For every exchange metabolite j and state s, species with F[s,i,j] > EPS are
donors and species with F[s,i,j] < -EPS are receivers. Flux from donors to
receivers is allocated proportionally (donor share x receiver uptake), which
gives a weighted donor -> receiver -> metabolite network per state.

Outputs (resultados/):
  intercambio_aristas.csv   donor, receiver, metabolite, scenario, mean flux,
                            fraction of states in which the link is active
  intercambio_especie.csv   per species: number of metabolites taken from the
                            community, from the medium only, and donated
  dependencias_<sp>.csv     metabolites each validated taxon obtains from
                            other species, with its main donors

Usage (Jupyter):  %run intercambio.py
"""
import pickle
import numpy as np
import pandas as pd
import pipeline_schubert as ps

EPS = 0.02
ESC = ("SS19", "SS37")
MEDIO = set(k + "_e" for k in ps.M9_GLICEROL) | {"glyc_e"}

mapa = pickle.load(open("mapa_todas.pkl", "rb"))
F = {e: np.load("cache/flujos_%s_n150.npz" % e)["F"] for e in ESC}
esp = mapa.especies
abv = [ps.NOMBRES.get(t, t) for t in esp]

filas = []
for e, Fe in F.items():
    n = len(Fe)
    acum = {}
    for s in range(n):
        X = Fe[s]                                   # species x metabolites
        for j in np.where((X > EPS).any(0) & (X < -EPS).any(0))[0]:
            don = np.where(X[:, j] > EPS)[0]
            rec = np.where(X[:, j] < -EPS)[0]
            tot = X[don, j].sum()
            for r in rec:
                for d in don:
                    f = -X[r, j] * X[d, j] / tot
                    k = (d, r, j)
                    a = acum.setdefault(k, [0.0, 0])
                    a[0] += f
                    a[1] += 1
    for (d, r, j), (f, c) in acum.items():
        filas.append({"escenario": e, "donante": esp[d], "receptor": esp[r],
                      "metabolito": mapa.met_e[j], "flujo_medio": f / n,
                      "frac_estados": c / n})
ar = pd.DataFrame(filas).sort_values("flujo_medio", ascending=False)
ar.to_csv("resultados/intercambio_aristas.csv", index=False)

# per-species summary (links active in >= 50% of the states of a scenario)
fuertes = ar[ar.frac_estados >= 0.5]
res = []
for t in esp:
    i = esp.index(t)
    tom = set(fuertes[fuertes.receptor == t].metabolito)
    don = set(fuertes[fuertes.donante == t].metabolito)
    activo = np.mean([np.mean(np.abs(F[e][:, i, :]).max(1) > EPS) for e in ESC])
    res.append({"especie": t, "activa_frac": activo,
                "recibe_n": len(tom), "recibe_no_medio": len(tom - MEDIO),
                "dona_n": len(don),
                "recibe": ";".join(sorted(m[:-2] for m in tom - MEDIO))})
res = pd.DataFrame(res)
res.to_csv("resultados/intercambio_especie.csv", index=False)
print(res.drop(columns="recibe").round(2).to_string(index=False))

for t in ("Salinibacter", "Haloquadratum"):
    d = (ar[(ar.receptor == t) & ~ar.metabolito.isin(MEDIO)]
         .groupby(["metabolito", "donante"])[["flujo_medio", "frac_estados"]].mean()
         .reset_index().sort_values("flujo_medio", ascending=False))
    d.to_csv("resultados/dependencias_%s.csv" % t, index=False)
    print("\n%s: top metabolites received from the community" % t)
    print(d.head(15).round(3).to_string(index=False))
