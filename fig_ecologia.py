# -*- coding: utf-8 -*-
"""
Fig. 8: cross-feeding in the curated community (from ecologia.py and
ecologia_dep.py results). Usage (Jupyter):  %run fig_ecologia.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figuras_revision as fr
import pipeline_schubert as ps

R = "resultados"
OUT = "figs"
fr.style()
AB = lambda t: ps.NOMBRES.get(t, t)
MET = {"dha": "dihydroxyacetone", "galct__D": "galactitol", "6pgc": "6-P-gluconate",
       "thr__L": "L-Thr", "lys__L": "L-Lys", "xan": "xanthine", "ura": "uracil",
       "arg__L": "L-Arg", "orn": "ornithine", "hxan": "hypoxanthine", "phe__L": "L-Phe",
       "met__L": "L-Met", "no3": "nitrate", "tyr__L": "L-Tyr", "gthrd": "glutathione",
       "trp__L": "L-Trp", "acald": "acetaldehyde", "gly": "glycine", "ile__L": "L-Ile",
       "man1p": "mannose 1-P", "s": "sulfur"}
nombre = lambda m: MET.get(m[:-2], m[:-2])

# growth in isolation (individual curation runs) and inside the community
iso = pd.read_csv("curacion_v2/v2_crecimiento_especies.csv").set_index("especie").crec_medio
cre = pd.read_csv(R + "/eco_crecimiento.csv").set_index("especie")
dep = pd.read_csv(R + "/eco_dependencias.csv")
imp = pd.read_csv(R + "/eco_hqr_importa.csv")

fig = plt.figure(figsize=(fr.W2, 4.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1, 1],
                      wspace=0.62, hspace=0.62)

# (A) growth alone vs in community -----------------------------------------
a = fig.add_subplot(gs[:, 0])
orden = cre.sort_values("max_en_comunidad").index.tolist()
y = np.arange(len(orden))
for i, t in enumerate(orden):
    x0, x1, x2 = iso.get(t, 0.0), cre.loc[t, "max_al_90"], cre.loc[t, "max_en_comunidad"]
    a.plot([min(x0, x1), max(x0, x2)], [i, i], color="#d6d5d0", lw=1.3, zorder=1)
a.scatter([iso.get(t, 0.0) for t in orden], y, s=20, color=fr.NEUTRAL, zorder=3,
          label="alone, M9 + glycerol")
a.scatter(cre.loc[orden, "max_al_90"], y, s=20, facecolor="white",
          edgecolor=fr.TIER_COLOR["T1"], lw=1.1, zorder=4,
          label="community, $\\geq$90% of optimum")
a.scatter(cre.loc[orden, "max_en_comunidad"], y, s=20, color=fr.TIER_COLOR["T2"], zorder=5,
          label="community, unconstrained")
lab = [AB(t) for t in orden]
a.set_yticks(y)
a.set_yticklabels(lab, fontstyle="italic", fontsize=6.4)
for tl, t in zip(a.get_yticklabels(), orden):
    if t in ("Salinibacter", "Haloquadratum"):
        tl.set_fontweight("bold")
a.set_xlabel("Maximal biomass flux (h$^{-1}$)")
a.set_xlim(-0.02, 0.68)
a.grid(axis="y", visible=False)
a.legend(loc="upper center", bbox_to_anchor=(0.42, -0.085), ncol=1, fontsize=6.1,
         handletextpad=0.3)
fr.panel_label(a, "A", dx=-78, dy=6)

# (B) Salinibacter: block each metabolite received ------------------------
b = fig.add_subplot(gs[0, 1])
s = dep[dep.especie == "Salinibacter"].copy()
s["lab"] = s.metabolito.map(nombre)
s = s.sort_values("max_libre")
col = [fr.NULL_COLOR if v < 1e-3 else fr.TIER_COLOR["T1"] for v in s.max_libre]
b.barh(np.arange(len(s)), s.max_libre, color=col, height=0.62)
b.axvline(cre.loc["Salinibacter", "max_en_comunidad"], color=fr.INK2, lw=0.8, ls="--")
b.set_yticks(np.arange(len(s)))
b.set_yticklabels(s.lab, fontsize=6.2)
b.set_xlabel("Max. biomass flux of $Sal.\\ ruber$ (h$^{-1}$)")
b.set_title("uptake from the community blocked", fontsize=6.6, color=fr.INK2, pad=3)
b.grid(axis="y", visible=False)
b.set_xlim(0, 0.5)
for i, v in enumerate(s.max_libre):
    if v < 1e-3:
        b.text(0.008, i, "no growth", va="center", fontsize=6.0, color=fr.NULL_COLOR,
               fontweight="bold")
fr.panel_label(b, "B", dx=-62, dy=6)

# (C) Haloquadratum at its community maximum: what it imports -------------
c = fig.add_subplot(gs[1, 1])
h = imp[~imp.en_medio].head(7).iloc[::-1]
dcol = {"Salinibacter": fr.TIER_COLOR["T2"], "Halomonas": "#eb6834",
        "Puniceicoccus": "#f2a541", "Gracilimonas": fr.TIER_COLOR["T0"]}
for i, (_, r) in enumerate(h.iterrows()):
    izq = 0.0
    partes = [p.split(":") for p in str(r.donantes).split(";") if ":" in p]
    tot = sum(float(v) for _, v in partes) or 1.0
    for dn, v in sorted(partes, key=lambda p: -float(p[1])):
        w = -r.flujo * float(v) / tot
        c.barh(i, w, left=izq, color=dcol.get(dn, "#c9c8c3"), height=0.62)
        izq += w
c.set_xscale("log")
c.set_xlim(0.01, 40)
c.set_yticks(np.arange(len(h)))
c.set_yticklabels([nombre(m) for m in h.metabolito], fontsize=6.2)
c.set_xlabel("Uptake by $Hqr.\\ walsbyi$ (mmol gDW$^{-1}$ h$^{-1}$, log)")
c.grid(axis="y", visible=False)
from matplotlib.patches import Patch
c.legend(handles=[Patch(color=v, label=AB(k)) for k, v in dcol.items()] +
         [Patch(color="#c9c8c3", label="other")],
         loc="lower right", fontsize=5.6, handlelength=0.9, handletextpad=0.3,
         labelspacing=0.25, prop={"style": "italic", "size": 5.6})
hq = dep[(dep.especie == "Haloquadratum") & (dep.metabolito == "dha_e")].iloc[0]
c.text(0.012, len(h) - 0.35, "DHA uptake blocked: max. at 90%% of optimum\n%.3f $\\rightarrow$ %.3f h$^{-1}$"
       % (cre.loc["Haloquadratum", "max_al_90"], hq.max_al_90), fontsize=5.6, color=fr.INK2,
       va="bottom")
c.set_ylim(-0.6, len(h) + 0.9)
fr.panel_label(c, "C", dx=-62, dy=6)
fr.save(fig, OUT, "fig8_crossfeeding")
print("written figs/fig8_crossfeeding")
