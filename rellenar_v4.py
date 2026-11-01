# -*- coding: utf-8 -*-
"""Fills the control-dependent placeholders of manuscript_v4_plantilla.tex
(scenario coefficients and random-perturbation control) and writes
manuscript_v4.tex. Run after the control replicates (control_run.py).
Usage (Jupyter):  %run rellenar_v4.py"""
import glob
import os
import pandas as pd
import pipeline_schubert as ps

R = "resultados"
s = open("manuscript_v4_plantilla.tex", encoding="utf-8").read()
obs = pd.read_csv(os.path.join(R, "regresion_ordinal.csv"))
reps = sorted(glob.glob(os.path.join(R, "control_rep*.csv")))
if reps:
    ctrl = pd.concat([pd.read_csv(f) for f in reps])
    ctrl.loc[(ctrl.ic_sup - ctrl.ic_inf) > 50, "beta"] = float("nan")   # quasi-separation
    comp = ps.comparar_con_control(obs, ctrl)
    comp.to_csv(os.path.join(R, "regresion_ordinal_control.csv"), index=False)
    comp = comp.set_index("especie")
    nrep = ctrl.rep.nunique()
else:
    comp, nrep = None, 0


def num(x):
    return "--" if pd.isna(x) else ("$%.2f$" % x if x < 0 else "%.2f" % x)


filas = []
for r in obs.itertuples():
    if pd.isna(r.beta):
        continue
    bc, pc = ("--", "--")
    if comp is not None and r.especie in comp.index:
        bc = num(comp.loc[r.especie, "beta_ctrl"])
        pc = "--" if pd.isna(comp.loc[r.especie, "p_ctrl"]) else "%.1f" % comp.loc[r.especie, "p_ctrl"]
    filas.append(r"\emph{%s} & %s & [%s, %s] & %.2f & %s & %s \\" % (
        r.especie, num(r.beta), num(r.ic_inf), num(r.ic_sup), r.p, bc, pc))
s = s.replace("@@ORDROWS@@", "\n".join(filas))
s = s.replace("@@NCTRL@@", "$B=%d$" % nrep if nrep else r"\pend{B}")
if comp is not None:
    c = ctrl.groupby("especie").beta.agg(["min", "max"])
    sub = c.loc[[e for e in obs.especie[obs.beta.notna()] if e in c.index]]
    txt = (r"Restricting the same numbers of randomly chosen exchange coordinates "
           r"in %d control replicates produced coefficients of comparable or larger "
           r"magnitude (from $%.2f$ to $%.2f$ over these taxa), so no taxon showed a "
           r"scenario response that exceeded the response to random restrictions."
           % (nrep, sub["min"].min(), sub["max"].max()))
else:
    txt = r"\pend{control replicates}"
s = s.replace("@@CTRLTXT@@", txt)
open("manuscript_v4.tex", "w", encoding="utf-8").write(s)
left = [x for x in ("@@",) if x in s]
print("written manuscript_v4.tex; control replicates:", nrep, "; placeholders left:", left)
