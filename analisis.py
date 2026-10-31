# -*- coding: utf-8 -*-
"""Runs every analysis of the revised manuscript on the cached samples."""
import json
import pickle
import numpy as np
import pandas as pd
import pipeline_schubert as ps
import paso1_codim_normalizada as p1

OUT = "resultados"
mapa = pickle.load(open("mapa_todas.pkl", "rb"))
Z = {e: np.load("cache/flujos_%s_n150.npz" % e) for e in ("SS19", "SS37")}
F = {e: Z[e]["F"] for e in Z}
cols = list(Z["SS19"]["cols"])

# tex map (Table 5 definition) from the same raw fluxes
mapa_tex = pickle.load(open("mapa_tex.pkl", "rb"))
F_tex = {e: ps.recalcular_F(Z[e]["V"], cols, mapa_tex) for e in Z}

# ---- tolerance (values used in the previous version; see manuscript Table 3)
tol = pd.DataFrame([
    ("Hqr. walsbyi", 14, 32, 1), ("Sal. ruber", 20, 30, 1), ("Hrr. salinarum", 10, 30, 1),
    ("Spb. salinus", 10, 25, 1), ("Hmc. mukohataei", 8.2, 19.9, 0),
    ("Arh. aquaeolei", 6, 20, 1), ("Pcc. vermicola", 1.0, 7.5, 1),
    ("Ocn. granulosus", 0.5, 10, 0), ("Gra. halophila", 1, 20, 0),
], columns=["especie", "min", "max", "alta_confianza"])
tol["tolerancia"] = tol["max"] - tol["min"]
tol.to_csv("tolerancia.csv", index=False)

res = {}
for etiqueta, mp, FF in (("todas", mapa, F), ("tex", mapa_tex, F_tex)):
    mu = ps.tabla_muestras(mp, FF, eps=1e-6, guardar="%s/muestras_%s.csv" % (OUT, etiqueta))
    esp, corr = p1.analizar(mu, tol, mp.m, "%s/paso1_%s" % (OUT, etiqueta), n_mc=200000)
    res[etiqueta] = dict(esp=esp, corr=corr)
    print("\n=== interface:", etiqueta, "===")
    print(esp.round(4).to_string())
    print(corr.round(4).to_string())

# ---- primary analyses on the recommended definition
mu = pd.read_csv("%s/muestras_todas.csv" % OUT)
print("\nrho_S(codim, d) over states:", round(pd.Series(mu.codim).corr(mu.d, method="spearman"), 3))
mrect = mu.d * (mapa.m - mu.d)
mu_null = mrect / 2
sd_null = np.sqrt(mrect * (mapa.m + 1) / 12)
print("fraction of states outside +-2SD random band:",
      round(float((np.abs(mu.codim - mu_null) > 2 * sd_null).mean()), 3))

sens, det = ps.sensibilidad_bandera(mapa, F, tol, B=1000, eps=1e-6,
                                    guardar="%s/sensibilidad_bandera_det.csv" % OUT)
sens.to_csv("%s/sensibilidad_bandera.csv" % OUT, index=False)
print("\nflag sensitivity\n", sens.round(3).to_string())

eps = ps.barrido_epsilon(mapa, F)
eps.to_csv("%s/barrido_epsilon.csv" % OUT, index=False)
print("\nmax |change nu| over eps:", round(eps.cambio_nu.abs().max(), 4))

ordn = ps.regresion_ordinal(mu)
ordn.to_csv("%s/regresion_ordinal.csv" % OUT, index=False)
print("\nordinal\n", ordn.round(3).to_string())

ph, D = ps.homologia_persistente(mapa, np.concatenate([F["SS19"], F["SS37"]]),
                                 guardar_bottleneck="%s/bottleneck.csv" % OUT)
ph = ph.merge(mu.groupby("especie")[["d", "nu"]].mean().reset_index(), on="especie")
ph.to_csv("%s/homologia_persistente.csv" % OUT, index=False)
print("\nPH\n", ph.round(3).to_string())
from scipy import stats
for c in ("persistencia_H0", "persistencia_H1"):
    g = ph.merge(tol, on="especie")
    print(c, "vs tol:", p1.spearman_perm(g[c], g["tolerancia"]),
          " vs nu:", p1.spearman_perm(ph[c], ph["nu"]))

json.dump({"m": mapa.m, "dims_todas": ps.dimensiones_bandera(mapa),
           "dims_tex": ps.dimensiones_bandera(mapa_tex)},
          open("%s/meta.json" % OUT, "w"), indent=1)
print("DONE ANALISIS")
