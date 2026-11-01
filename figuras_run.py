# -*- coding: utf-8 -*-
"""Manuscript figures from the real run (resultados/)."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figuras_revision as fr

OUT = "figs"
os.makedirs(OUT, exist_ok=True)
fr.style()
meta = json.load(open("resultados/meta.json"))
M = meta["m"]
tol = pd.read_csv("tolerancia.csv")

fr.fig_schema(OUT)
fr.fig_diagnostic(OUT)
dfs = fr.load_samples("resultados/muestras_todas.csv", None, M)
val = set(fr.TIER)
fr.fig_nu_species(dfs[dfs.especie.isin(val)], OUT)
fr.fig_h1a(dfs[dfs.especie.isin(val)], tol, M, OUT)
fr.fig_bottleneck(pd.read_csv("resultados/bottleneck.csv", index_col=0), OUT)

# ---- Fig. 6: what the reference ordering contributes
det = pd.read_csv("resultados/sensibilidad_bandera_det.csv")
var = pd.read_csv("resultados/nu_por_variante.csv")
fig = plt.figure(figsize=(fr.W2, 3.0))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.75, wspace=0.42)
a = fig.add_subplot(gs[:, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, 1])
orden = fr.species_order(list(var.especie))
y = np.arange(len(orden))
estilos = [("original", "Santa Pola flag", fr.TIER_COLOR["T2"], "o", True),
           ("solo_frecuencia", "frequency only (no tiers)", fr.TIER_COLOR["T1"], "D", False),
           ("invertida", "tiers reversed", "#eb6834", "^", False),
           ("aleatoria", "random flags (mean)", fr.NEUTRAL, "s", True)]
v = var.set_index("especie").loc[orden]
for col, lab, colr, mk, lleno in estilos:
    a.plot(v[col], y, mk, ms=4.2, color=colr, mfc=colr if lleno else "white", mew=1.0,
           ls="", label=lab)
a.axvline(0.5, color=fr.NULL_COLOR, lw=0.7, ls=(0, (3, 2)))
a.set_yticks(y); a.set_yticklabels(orden, fontstyle="italic"); a.invert_yaxis()
a.set_xlabel(r"Mean normalized codimension $\nu$")
a.set_xlim(0.45, 1.0)
a.grid(axis="y", visible=False)
a.legend(loc="lower center", bbox_to_anchor=(0.45, -0.30), ncol=2, fontsize=6.0,
         handletextpad=0.2, columnspacing=0.8)
fr.panel_label(a, "A", dx=-70, dy=4)
bins = np.linspace(-1, 1, 41)
b.hist(det[det.variante == "aleatoria"].rho, bins=bins, color="#cfcfca", edgecolor="white",
       linewidth=0.6, label="random flags")
b.hist(det[det.variante == "permutada"].rho, bins=bins, color=fr.TIER_COLOR["T1"], alpha=0.75,
       edgecolor="white", linewidth=0.6, label="within-tier permutations")
r0 = det[det.variante == "original"].rho.iloc[0]
b.axvline(r0, color=fr.INK, lw=1.1, label="observed")
b.set_xlabel(r"$\rho_S$ (mean $\nu$ vs NaCl tolerance)")
b.set_ylabel("References")
b.legend(loc="upper left", fontsize=5.6, framealpha=0.9)
fr.panel_label(b, "B", dy=4)
c.hist(det[det.variante == "aleatoria"].tau, bins=bins, color="#cfcfca", edgecolor="white",
       linewidth=0.6)
c.hist(det[det.variante == "permutada"].tau, bins=bins, color=fr.TIER_COLOR["T1"], alpha=0.75,
       edgecolor="white", linewidth=0.6)
ti = det[det.variante == "invertida"].tau.iloc[0]
c.axvline(ti, color="#eb6834", lw=1.1, ls=(0, (3, 2)), label="tiers reversed")
c.legend(loc="upper left", fontsize=5.8)
c.set_xlabel(r"Kendall $\tau$ with the original species ranking")
c.set_ylabel("References")
fr.panel_label(c, "C", dy=4)
fr.save(fig, OUT, "fig6_flag_sensitivity")

# ---- Fig. S2: synthetic study
if os.path.exists("resultados/sintetico.csv"):
    s = pd.read_csv("resultados/sintetico.csv", keep_default_na=False)
    fig, axes = plt.subplots(1, 3, figsize=(fr.W2, 2.1), sharey=True,
                             gridspec_kw=dict(wspace=0.12))
    titulos = {"positional": "Positional signal", "confounded": "Confounded by $d$",
               "null": "Null"}
    for k, (ax, reg) in enumerate(zip(axes, ("positional", "confounded", "null"))):
        t = s[s.regimen == reg]
        ax.plot(t.n_s, t.rechazo_raw, "o-", color=fr.NEUTRAL, ms=4, lw=1.2,
                label=r"raw $|\lambda|$")
        ax.plot(t.n_s, t.rechazo_nu, "s-", color=fr.TIER_COLOR["T1"], ms=4, lw=1.2,
                label=r"normalized $\nu$")
        ax.axhline(0.05, color=fr.NULL_COLOR, lw=0.7, ls=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_xticks([4, 6, 8, 11, 20, 50])
        ax.set_xticklabels(["4", "6", "8", "11", "20", "50"])
        ax.minorticks_off()
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Number of species $n_s$")
        ax.set_title(titulos[reg], loc="left", fontsize=7)
        fr.panel_label(ax, "ABC"[k], dy=12)
    axes[0].set_ylabel(r"Rejection rate ($\alpha=0.05$)")
    axes[0].legend(loc="center right", fontsize=6.2)
    fr.save(fig, OUT, "figS2_synthetic")
print("figures:", sorted(os.listdir(OUT)))
